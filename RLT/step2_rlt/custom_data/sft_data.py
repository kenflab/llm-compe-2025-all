from datasets import load_dataset, concatenate_datasets
from .utils import make_masked_sft_collator
from .reasoning_datasets_info import (
    DATA_CONFIGS, wrap_string_between_tag, grab_text_between_tag, get_tags,
    ReasoningData)


def add_indices(ds):
    if "__index" not in ds.column_names:
        ds = ds.map(lambda x, i: {"__index": i}, with_indices=True)
    return ds


def get_process_line_fn(datasets):
    """複数データセット対応版のprocess_line_fn生成関数"""
    # 後方互換性のため、単一データセットの場合の処理（文字列で渡された場合）
    if isinstance(datasets, str):
        dataset_id_or_path = datasets
        data: ReasoningData = DATA_CONFIGS[dataset_id_or_path]
        system_prompt = data.system_prompt

        def process_line_fn(line, tokenizer):
            question_content, thought_process_and_solution = (
                data.extract_question_and_completion_from_line(line))
            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": question_content,
                },
                {
                    "role": "assistant",
                    "content": thought_process_and_solution,
                }
            ]
            line_text = tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=False)
            return {"text": line_text}
        return process_line_fn
    
    # 複数データセットの場合
    # 全データセットが同じ形式なので、最初のデータセットのDATA_CONFIGSエントリーを使用
    # もしくは、bespokelabs/Bespoke-Stratos-17kのエントリーを使用
    base_dataset_id = "bespokelabs/Bespoke-Stratos-17k"
    
    # DATA_CONFIGSからデータ処理クラスを取得
    if base_dataset_id in DATA_CONFIGS:
        data: ReasoningData = DATA_CONFIGS[base_dataset_id]
    else:
        # もし見つからない場合は、最初のデータセットIDを試す
        first_dataset_id = datasets[0]["dataset_id_or_path"] if isinstance(datasets, list) else datasets
        data: ReasoningData = DATA_CONFIGS[first_dataset_id]
    
    system_prompt = data.system_prompt
    
    def process_line_fn(line, tokenizer):
        question_content, thought_process_and_solution = (
            data.extract_question_and_completion_from_line(line))
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question_content,
            },
            {
                "role": "assistant",
                "content": thought_process_and_solution,
            }
        ]
        line_text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=False)
        return {"text": line_text}
    
    return process_line_fn


def load_formatted_sft_dataset(
        tokenizer,
        dataset_id_or_path=None,  # 後方互換性のためデフォルトをNoneに
        datasets=None,  # 新規追加：複数データセット対応
        dataset_local_directory=None,
        train_split='train',
        val_split=None,
        process_line_fn=None,
        model_name_or_path=None,
        completion_only_training=True,
        custom_start_of_response=None,
        keep_columns=None,
        add_dataset_indices=False,
        artificial_epochs=None,
        **dataset_loading_kwargs,
):
    # 後方互換性のため、dataset_id_or_pathが指定された場合の処理
    if dataset_id_or_path is not None:
        datasets = [{
            "dataset_id_or_path": dataset_id_or_path,
            "dataset_configs": dataset_loading_kwargs.get("dataset_configs", None),
            "num_samples": None
        }]
    
    # datasetsがNoneの場合のエラーチェック
    if datasets is None:
        raise ValueError("Either 'datasets' or 'dataset_id_or_path' must be provided")
    
    print(f"\n{'='*80}")
    print(f"Loading {len(datasets)} dataset(s)")
    print(f"{'='*80}")
    
    # 複数データセットを読み込んでマージ
    all_train_datasets = []
    all_val_datasets = []
    dataset_stats = []  # 統計情報を保存
    
    for idx, dataset_info in enumerate(datasets, 1):
        current_dataset_id = dataset_info["dataset_id_or_path"]
        num_samples = dataset_info.get("num_samples", None)
        
        print(f"\n[Dataset {idx}/{len(datasets)}] Loading: {current_dataset_id}")
        print(f"  - Requested samples: {num_samples if num_samples else 'All'}")
        
        # データセット読み込み（元のソースと同じように、dataset_loading_kwargsをそのまま使用）
        current_local_dir = dataset_local_directory or current_dataset_id
        dataset = load_dataset(current_local_dir, **dataset_loading_kwargs)
        
        # 訓練データセット処理
        train_dataset = dataset[train_split]
        original_train_size = len(train_dataset)
        
        # 件数制限の適用
        if num_samples is not None and num_samples < len(train_dataset):
            train_dataset = train_dataset.shuffle(seed=42).select(range(num_samples))
            print(f"  - Train data: {original_train_size} → {len(train_dataset)} samples (limited)")
        else:
            print(f"  - Train data: {len(train_dataset)} samples (all)")
        
        # データセットIDを追加（複数データセットの識別用）
        train_dataset = train_dataset.map(
            lambda x: {"__dataset_id": current_dataset_id}
        )
        
        all_train_datasets.append(train_dataset)
        
        # 検証データセット処理
        val_size = 0
        if val_split is not None:
            val_dataset = dataset[val_split]
            original_val_size = len(val_dataset)
            
            if num_samples is not None:
                # 検証データは訓練データの10%程度を目安に
                val_samples = max(1, num_samples // 10)
                if val_samples < len(val_dataset):
                    val_dataset = val_dataset.shuffle(seed=42).select(range(val_samples))
                    print(f"  - Val data: {original_val_size} → {len(val_dataset)} samples (limited)")
                else:
                    print(f"  - Val data: {len(val_dataset)} samples (all)")
            else:
                print(f"  - Val data: {len(val_dataset)} samples (all)")
            
            val_size = len(val_dataset)
            val_dataset = val_dataset.map(
                lambda x: {"__dataset_id": current_dataset_id}
            )
            all_val_datasets.append(val_dataset)
        
        # 統計情報を保存
        dataset_stats.append({
            "dataset": current_dataset_id,
            "train_samples": len(train_dataset),
            "val_samples": val_size
        })
    
    # 複数データセットをマージしてシャッフル
    train_dataset = concatenate_datasets(all_train_datasets)
    train_dataset = train_dataset.shuffle(seed=42)
    
    print(f"\n{'='*80}")
    print(f"Dataset Loading Summary:")
    print(f"{'='*80}")
    for stat in dataset_stats:
        print(f"  {stat['dataset']}:")
        print(f"    - Train: {stat['train_samples']:,} samples")
        if stat['val_samples'] > 0:
            print(f"    - Val: {stat['val_samples']:,} samples")
    
    print(f"\nTotal merged dataset:")
    print(f"  - Train: {len(train_dataset):,} samples")
    
    # データセットごとの分布を確認
    if len(datasets) > 1:
        print(f"\nDataset distribution in merged train set:")
        dataset_counts = {}
        for item in train_dataset:
            ds_id = item.get("__dataset_id", "unknown")
            dataset_counts[ds_id] = dataset_counts.get(ds_id, 0) + 1
        
        for ds_id, count in dataset_counts.items():
            percentage = (count / len(train_dataset)) * 100
            print(f"  - {ds_id}: {count:,} samples ({percentage:.1f}%)")
    
    if add_dataset_indices:
        train_dataset = add_indices(train_dataset)
    
    # process_line_fnの適用
    if process_line_fn is not None:
        if isinstance(process_line_fn, (list, tuple)):
            processed_train_datasets = []
            for fn in process_line_fn:
                processed = train_dataset.map(
                    lambda x, fn=fn: fn(x, tokenizer))
                processed_train_datasets.append(processed)
            train_dataset = concatenate_datasets(
                processed_train_datasets)
        else:
            print('not loading from cache')
            train_dataset = train_dataset.map(
                lambda x: process_line_fn(x, tokenizer))
    
    # 検証データセット処理
    if val_split is None or len(all_val_datasets) == 0:
        val_dataset = None
        print(f"  - Val: No validation set")
    else:
        val_dataset = concatenate_datasets(all_val_datasets)
        val_dataset = val_dataset.shuffle(seed=42)
        print(f"  - Val: {len(val_dataset):,} samples")
        
        if add_dataset_indices:
            val_dataset = add_indices(val_dataset)
        
        if process_line_fn is not None:
            if isinstance(process_line_fn, (list, tuple)):
                processed_val_datasets = []
                for fn in process_line_fn:
                    processed = val_dataset.map(
                        lambda x, fn=fn: fn(x, tokenizer))
                    processed_val_datasets.append(processed)
                val_dataset = concatenate_datasets(
                    processed_val_datasets)
            else:
                val_dataset = val_dataset.map(
                    lambda x: process_line_fn(x, tokenizer))
    
    print(f"{'='*80}\n")
    
    if keep_columns is not None:
        # __dataset_idは保持
        keep_cols = keep_columns + ["__dataset_id"] if "__dataset_id" not in keep_columns else keep_columns
        train_dataset = train_dataset.remove_columns(
            [col for col in train_dataset.column_names
             if col not in keep_cols])
    
    if artificial_epochs is not None:
        assert artificial_epochs == 1, (
            'Artificial epoch, moved to GRPO to avoid shuffling samples between'
            ' different epochs.')
        
        train_dataset = concatenate_datasets(
            [train_dataset]*artificial_epochs)
    
    out_data = dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    if completion_only_training:
        out_data['data_collator'] = make_masked_sft_collator(
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            custom_start_of_response=custom_start_of_response,
        )
    return out_data