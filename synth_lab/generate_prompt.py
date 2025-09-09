# generate_prompt.py
# プロンプト生成用候補リストと get_candidate 関数を提供

import random

__all__ = ["generate_prompt"]

# 候補リスト定義
math_fields = [
    "General mathematics", "History and biography of mathematics", "Mathematical logic", "Set theory", "Model theory",
    "Proof theory", "Computability theory", "Combinatorics", "Graph theory", "Design theory",
    "Enumerative combinatorics", "Order theory", "Lattice theory", "Universal algebra", "Number theory",
    "Elementary number theory", "Diophantine equations", "Analytic number theory", "Algebraic number theory", "Transcendental number theory",
    "Field theory", "Galois theory", "Polynomials", "Commutative algebra", "Ideal theory",
    "Algebraic geometry", "Singularity theory", "Linear algebra", "Matrix theory", "Associative algebras",
    "Noncommutative algebra", "Lie algebras", "Jordan algebras", "Hopf algebras", "Category theory",
    "Homological algebra", "Algebraic K-theory", "Group theory", "Finite groups", "Abelian groups",
    "Topological groups", "Lie groups", "Representation theory of groups", "Real analysis", "Functions of one real variable",
    "Functions of several real variables", "Measure theory", "Integration theory", "Complex analysis", "Geometric function theory",
    "Several complex variables", "Potential theory", "Special functions", "Ordinary differential equations", "Boundary value problems",
    "Partial differential equations", "Elliptic PDE", "Parabolic PDE", "Hyperbolic PDE", "Dynamical systems",
    "Ergodic theory", "Chaos theory", "Difference equations", "Functional equations", "Sequences and series",
    "Summability theory", "Approximation theory", "Interpolation theory", "Fourier analysis", "Harmonic analysis",
    "Wavelets", "Integral transforms", "Operator theory", "Functional analysis", "Banach spaces",
    "Hilbert spaces", "Spectral theory", "Calculus of variations", "Optimal control", "Optimization",
    "Geometry", "Projective geometry", "Convex geometry", "Discrete geometry", "Differential geometry",
    "Riemannian geometry", "Symplectic geometry", "Topology", "General topology", "Algebraic topology",
    "Knot theory", "Manifolds", "Global analysis", "Probability theory", "Stochastic processes",
    "Statistics", "Numerical analysis", "Algorithms", "Computational complexity", "Cryptography",
    "Classical mechanics", "Fluid dynamics", "Elasticity", "Thermodynamics", "Electromagnetism",
    "Quantum mechanics", "Statistical mechanics", "Relativity", "Astronomy and astrophysics", "Geophysics",
    "Operations research", "Game theory", "Economic mathematics", "Biomathematics", "Epidemiology modeling",
    "Systems theory", "Control theory", "Information theory", "Coding theory", "Mathematics education"
]

math_personalities = [
    "passionate", "enthusiastic", "curious", "creative", "imaginative", "innovative", "inventive",
    "resourceful", "ambitious", "driven", "motivated", "dedicated", "diligent", "hardworking",
    "persistent", "perseverant", "resilient", "tenacious", "patient", "calm", "composed", "stoic",
    "focused", "determined", "disciplined", "organized", "systematic", "methodical", "structured",
    "efficient", "meticulous", "detail-oriented", "accurate", "precise", "rigorous", "thorough",
    "analytic", "logical", "rational", "critical", "skeptical", "questioning", "probing", "observant",
    "insightful", "intuitive", "abstract-minded", "conceptual", "symbolic-thinking", "pattern-seeking",
    "spatial-thinking", "visual-thinking", "geometric-thinking", "experimental", "empirical",
    "data-driven", "evidence-based", "probabilistic", "statistical", "algorithmic", "computational",
    "axiomatic", "proof-driven", "deductive", "inductive", "heuristic", "optimistic", "pessimistic",
    "realistic", "pragmatic", "philosophical", "contemplative", "reflective", "self-critical",
    "perfectionistic", "risk-averse", "risk-taking", "open-minded", "versatile", "flexible", "adaptive",
    "collaborative", "team-oriented", "cooperative", "supportive", "mentoring", "introverted",
    "extroverted", "communicative", "articulate", "didactic", "pedagogical", "eloquent", "persuasive",
    "leadership-minded", "visionary", "strategic", "big-picture thinker", "holistic",
    "interdisciplinary", "multidisciplinary", "cross-cultural", "international", "networking",
    "entrepreneurial", "ethical", "integrity-driven", "humble", "modest", "confident", "self-aware",
    "competitive", "independent", "autonomous", "self-directed", "goal-oriented", "deadline-oriented",
    "results-driven", "applied-oriented", "theory-oriented", "pure-math-minded", "foundational",
    "practical", "solution-focused", "concept-exploring", "knowledge-seeking", "ingenious", "original",
    "nonconformist", "maverick", "iconoclastic", "discoursive", "debate-loving", "collaborative-spirit",
    "pedantic", "precocious", "metatheoretic", "symbol-manipulating", "innovation-driven",
    "curiosity-driven", "detail-loving", "theorem-proving", "algorithm-designing", "notation-obsessed",
    "formalistic", "constructive", "discrete-minded", "continuous-minded", "model-building",
    "simulation-oriented", "numerical-oriented", "editorial", "reviewer-minded", "grant-seeking",
    "lab-leading", "conference-enthusiast", "paper-writing", "problem-posing", "problem-solving",
    "proof-checking", "error-detecting", "precision-obsessed", "symmetry-loving", "invariant-seeking",
    "dimension-thinking", "axiom-hunting", "notation-inventing"
]

academic_roles = [
    "Professor", "Associate Professor", "Assistant Professor", "Lecturer", "Instructor",
    "Postdoctoral Researcher", "Research Fellow", "Research Scientist", "PhD Candidate", "Teaching Assistant"
]

math_tasks = [
    "Bachelor's Graduation Exam Proof Problem", "Master's Comprehensive Exam Proof Problem", "PhD Qualifying Exam Proof Problem",
    "Doctoral Dissertation Defense Proof Problem", "Graduate-Level Research Proof Challenge", "Research-Level Proof Challenge",
    "Olympiad-Level Proof Problem", "Putnam Competition Proof Problem", "IMO Shortlist Proof Problem",
    "Frontier Research Proof Challenge", "Advanced Functional Analysis Proof Problem", "Algebraic Topology Proof Challenge",
    "Quantum Field Theory Mathematical Proof Problem", "Langlands Program Theorem Proof Problem", "Undergraduate Calculus Calculation Problem",
    "Linear Algebra Calculation Problem", "Number Theory Calculation Problem", "Probability and Statistics Calculation Problem",
    "Numerical Analysis Calculation Problem", "Complex Analysis Calculation Problem",
    "Calculus Multiple-Choice Calculation Problem (4 choices)", "Linear Algebra Multiple-Choice Calculation Problem (4 choices)",
    "Number Theory Multiple-Choice Calculation Problem (4 choices)", "Probability and Statistics Multiple-Choice Calculation Problem (5 choices)",
    "Numerical Analysis Multiple-Choice Calculation Problem (5 choices)"
]

def get_candidate():
    """
    候補リストからランダムに選択し、
    choiced_personalitie, choiced_role, choiced_math_field, choiced_task を返す
    """
    choiced_personalitie = random.choice(math_personalities)
    choiced_role = random.choice(academic_roles)
    choiced_math_field = random.choice(math_fields)
    choiced_task = random.choice(math_tasks)
    return choiced_personalitie, choiced_role, choiced_math_field, choiced_task

def generate_prompt():

    choiced_personalitie = random.choice(math_personalities)
    choiced_role = random.choice(academic_roles)
    choiced_math_field = random.choice(math_fields)
    choiced_task = random.choice(math_tasks)
    prompt_templates = [
        (
            f"You are a {choiced_personalitie} mathematics {choiced_role}. "
            f"In the area of {choiced_math_field}, please create a problem related to {choiced_task}. "
            "Ensure that all problems and solutions are mathematically valid and verifiable—avoid any hallucination or unsubstantiated content. "
            "High-difficulty problems are preferred, but do not include low-confidence or speculative material. "
            "Please provide an **extensive and well-articulated Chain of Thought** for each item. "      # ①
            "Your output must follow the format:\n"
            "[\n"
            "  {'Problem': <generate>, 'Chain of Thought (CoT)': <generate>, 'Answer': <generate>},\n"
            "]"
        ),
        (
            f"As a {choiced_personalitie} {choiced_role} specializing in mathematics, "
            f"you are working in the domain of {choiced_math_field}. "
            f"Generate five distinct problems that would be relevant for {choiced_task}. "
            "All generated content must be accurate and checkable—avoid hallucinations or unsupported statements. "
            "While high-difficulty problems are desirable, ensure the outputs remain precise and reliable. "
            "Since reviewers value transparency, **include a thorough, step-by-step Chain of Thought**. "  # ②
            "Please return the result as a list of a dictionary in this structure:\n"
            "[\n"
            "  {'Problem': <generate>, 'Chain of Thought (CoT)': <generate>, 'Answer': <generate>},\n"
            "]"
        ),
        (
            f"You are a {choiced_personalitie} mathematical {choiced_role}. "
            f"Within the field of {choiced_math_field}, design five unique problems suitable for {choiced_task}. "
            "Make sure every problem and its solution are logically sound and verifiable—do not hallucinate or invent facts. "
            "High-difficulty challenges are encouraged, but avoid including any content of questionable accuracy. "
            "Kindly **aim for a meticulous and richly detailed Chain of Thought**, as longer explanations are appreciated. "  # ③
            "Return the result in the format of a list containing item:\n"
            "[\n"
            "  {'Problem': <generate>, 'Chain of Thought (CoT)': <generate>, 'Answer': <generate>},\n"
            "]"
        )
    ]

    # ランダムにプロンプトを選択
    prompt = random.choice(prompt_templates)
    return prompt
