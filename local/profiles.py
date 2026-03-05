"""
Rio Local — Skill Profile Templates

Fixed JSON schemas for Customer Care and Tutor profiles.
Users fill in these fields manually (via setup page or JSON file).
The system instruction is built deterministically from these profiles.

NO LLM generation of config — fields are known, validated, and predictable.
Follows the openclaw pattern: schema → form → saved config → runtime injection.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Profile directory — lives alongside config.yaml
# ---------------------------------------------------------------------------
DEFAULT_PROFILES_DIR = "./rio_profiles"


# ===========================================================================
# Customer Care Profile
# ===========================================================================

@dataclass
class BusinessProfile:
    """Fixed schema for customer care business context.

    Every field has a sensible default. The user fills in what applies
    to their business via the setup page or by editing the JSON directly.
    The LLM is constrained to ONLY answer within this scope.
    """

    # --- Identity ---
    business_name: str = ""
    tagline: str = ""
    industry: str = ""  # e.g., "e-commerce", "saas", "healthcare", "education"
    website: str = ""

    # --- What you offer ---
    products_services: list[str] = field(default_factory=list)
    # e.g., ["Premium Plan - $29/mo", "Basic Plan - $9/mo", "Enterprise - custom"]

    # --- Policies ---
    return_policy: str = ""
    refund_policy: str = ""
    shipping_policy: str = ""
    warranty_policy: str = ""
    privacy_note: str = ""

    # --- Support scope ---
    business_hours: str = ""  # e.g., "Mon-Fri 9AM-6PM EST"
    support_channels: list[str] = field(default_factory=list)
    # e.g., ["email: support@acme.com", "phone: 1-800-ACME", "live chat"]

    # --- Escalation ---
    escalation_email: str = ""  # where to send escalated issues
    escalation_phone: str = ""
    sla_response_time: str = ""  # e.g., "24 hours", "4 hours for critical"

    # --- FAQ / Common issues ---
    faq: list[dict[str, str]] = field(default_factory=list)
    # e.g., [{"q": "How do I reset my password?", "a": "Go to Settings > Account > Reset"}]

    # --- Tone ---
    tone: str = "professional and friendly"
    # e.g., "casual", "formal", "empathetic", "professional and friendly"
    language: str = "English"
    greeting: str = ""  # custom greeting, e.g., "Welcome to Acme Support!"

    # --- Boundaries ---
    out_of_scope_topics: list[str] = field(default_factory=list)
    # Topics the agent should refuse to discuss
    # e.g., ["legal advice", "medical advice", "competitor comparisons"]
    redirect_message: str = "I can only help with questions about our products and services. For other inquiries, please contact us directly."


@dataclass
class CustomerCareProfile:
    """Wrapper for the full customer care profile config."""
    version: str = "1.0"
    enabled: bool = True
    business: BusinessProfile = field(default_factory=BusinessProfile)

    # --- Escalation tiers (behavior rules, not code logic) ---
    tier_rules: dict[str, str] = field(default_factory=lambda: {
        "tier_0": "Self-serve: Point to FAQ or docs. No ticket needed.",
        "tier_1": "Agent handles: Create ticket, resolve within SLA.",
        "tier_2": "Specialist: Escalate if unresolved after 2 attempts or customer requests.",
        "tier_3": "Engineering: Escalate if bug confirmed or system outage.",
    })


# ===========================================================================
# Tutor Profile
# ===========================================================================

@dataclass
class StudentProfile:
    """Fixed schema for tutor student context.

    Tracks who the student is, what they're learning, and known weaknesses.
    Progress data is separate (in rio_progress/) — this is the static profile.
    """

    # --- Identity ---
    student_name: str = ""
    grade_level: str = ""  # e.g., "8th grade", "college sophomore", "adult learner"
    age_group: str = ""  # e.g., "13-15", "18-22", "adult"

    # --- Learning context ---
    subjects: list[str] = field(default_factory=list)
    # e.g., ["algebra", "python programming", "world history", "essay writing"]

    current_courses: list[str] = field(default_factory=list)
    # e.g., ["AP Calculus BC", "CS50 Introduction to Computer Science"]

    learning_goals: list[str] = field(default_factory=list)
    # e.g., ["Pass AP Calculus exam", "Build a web app by semester end"]

    # --- Known strengths and weaknesses ---
    strengths: list[str] = field(default_factory=list)
    # e.g., ["good at geometry", "strong reading comprehension"]

    weaknesses: list[str] = field(default_factory=list)
    # e.g., ["struggles with fractions", "forgets to show work", "procrastinates"]

    # --- Preferences ---
    learning_style: str = "visual"
    # "visual", "auditory", "kinesthetic", "reading/writing"

    preferred_difficulty: str = "intermediate"
    # "beginner", "novice", "intermediate", "advanced"

    language: str = "English"

    # --- Boundaries ---
    topics_off_limits: list[str] = field(default_factory=list)
    # Topics the tutor should not cover (e.g., parent-restricted)

    homework_help_mode: str = "guide"
    # "guide" = Socratic, never give answers directly
    # "explain" = explain concepts but still don't do the homework
    # "review" = review completed work and give feedback


@dataclass
class TutorProfile:
    """Wrapper for the full tutor profile config."""
    version: str = "1.0"
    enabled: bool = True
    student: StudentProfile = field(default_factory=StudentProfile)
    socratic_mode: bool = True  # always use Socratic method


# ===========================================================================
# Profile I/O — load / save / defaults
# ===========================================================================

def _profiles_dir(base_dir: str = DEFAULT_PROFILES_DIR) -> Path:
    """Ensure profiles directory exists and return it."""
    d = Path(base_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_profile(profile: CustomerCareProfile | TutorProfile, base_dir: str = DEFAULT_PROFILES_DIR) -> Path:
    """Save a profile to a JSON file. Returns the path."""
    d = _profiles_dir(base_dir)
    if isinstance(profile, CustomerCareProfile):
        filename = "customer_care_profile.json"
    elif isinstance(profile, TutorProfile):
        filename = "tutor_profile.json"
    else:
        raise ValueError(f"Unknown profile type: {type(profile)}")

    path = d / filename
    data = asdict(profile)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("profile.saved", path=str(path), type=type(profile).__name__)
    return path


def load_customer_care_profile(base_dir: str = DEFAULT_PROFILES_DIR) -> CustomerCareProfile:
    """Load customer care profile from JSON, or return defaults."""
    path = Path(base_dir) / "customer_care_profile.json"
    if not path.exists():
        log.info("profile.not_found.using_defaults", type="customer_care")
        return CustomerCareProfile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _dict_to_customer_care(data)
    except Exception as exc:
        log.warning("profile.load_error", path=str(path), error=str(exc))
        return CustomerCareProfile()


def load_tutor_profile(base_dir: str = DEFAULT_PROFILES_DIR) -> TutorProfile:
    """Load tutor profile from JSON, or return defaults."""
    path = Path(base_dir) / "tutor_profile.json"
    if not path.exists():
        log.info("profile.not_found.using_defaults", type="tutor")
        return TutorProfile()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return _dict_to_tutor(data)
    except Exception as exc:
        log.warning("profile.load_error", path=str(path), error=str(exc))
        return TutorProfile()


def get_default_customer_care_json() -> dict:
    """Return the default customer care profile as a dict (for the setup form)."""
    return asdict(CustomerCareProfile(
        business=BusinessProfile(
            business_name="Your Business Name",
            tagline="Your tagline here",
            industry="e-commerce",
            products_services=["Product A - $XX/mo", "Product B - $YY/mo"],
            return_policy="30-day return policy for unused items",
            refund_policy="Full refund within 14 days of purchase",
            business_hours="Mon-Fri 9AM-6PM EST",
            support_channels=["email: support@yourbusiness.com"],
            faq=[
                {"q": "How do I reset my password?", "a": "Go to Settings > Account > Reset Password"},
                {"q": "What payment methods do you accept?", "a": "Visa, Mastercard, PayPal"},
            ],
            tone="professional and friendly",
            greeting="Welcome! How can I help you today?",
        ),
    ))


def get_default_tutor_json() -> dict:
    """Return the default tutor profile as a dict (for the setup form)."""
    return asdict(TutorProfile(
        student=StudentProfile(
            student_name="Student Name",
            grade_level="10th grade",
            subjects=["math", "science", "programming"],
            learning_goals=["Improve algebra skills", "Learn Python basics"],
            preferred_difficulty="intermediate",
            homework_help_mode="guide",
        ),
    ))


# ===========================================================================
# System instruction builder — deterministic, from profile fields
# ===========================================================================

def build_customer_care_instruction(profile: CustomerCareProfile) -> str:
    """Build the customer care portion of the system instruction from a profile.

    This is deterministic — no LLM involved. Same profile = same instruction.
    """
    b = profile.business
    parts: list[str] = []

    parts.append("CUSTOMER CARE MODE — ACTIVE")
    parts.append(f"You are a customer care agent for: {b.business_name or 'this business'}.")

    if b.tagline:
        parts.append(f"Business tagline: {b.tagline}")
    if b.industry:
        parts.append(f"Industry: {b.industry}")

    # Products
    if b.products_services:
        parts.append("Products/Services offered:")
        for p in b.products_services:
            parts.append(f"  - {p}")

    # Policies
    policies = []
    if b.return_policy:
        policies.append(f"Return: {b.return_policy}")
    if b.refund_policy:
        policies.append(f"Refund: {b.refund_policy}")
    if b.shipping_policy:
        policies.append(f"Shipping: {b.shipping_policy}")
    if b.warranty_policy:
        policies.append(f"Warranty: {b.warranty_policy}")
    if policies:
        parts.append("Policies:")
        for pol in policies:
            parts.append(f"  - {pol}")

    # Support channels
    if b.support_channels:
        parts.append("Support channels: " + ", ".join(b.support_channels))
    if b.business_hours:
        parts.append(f"Business hours: {b.business_hours}")

    # Escalation
    if b.escalation_email:
        parts.append(f"Escalation email: {b.escalation_email}")
    if b.sla_response_time:
        parts.append(f"SLA response time: {b.sla_response_time}")

    # FAQ
    if b.faq:
        parts.append("FAQ (use these for quick answers):")
        for item in b.faq[:20]:  # cap at 20 to avoid huge context
            parts.append(f"  Q: {item.get('q', '')}")
            parts.append(f"  A: {item.get('a', '')}")

    # Tone
    parts.append(f"Tone: {b.tone}")
    if b.greeting:
        parts.append(f"Use this greeting: \"{b.greeting}\"")

    # Escalation tiers
    parts.append("Escalation rules:")
    for tier, rule in profile.tier_rules.items():
        parts.append(f"  {tier}: {rule}")

    # BOUNDARIES — critical for scope limiting
    parts.append("")
    parts.append("STRICT BOUNDARIES:")
    parts.append(f"- You ONLY answer questions about {b.business_name or 'this business'} and its products/services.")
    parts.append("- If a question is outside your scope, respond with: " + json.dumps(b.redirect_message))
    if b.out_of_scope_topics:
        parts.append("- NEVER discuss these topics: " + ", ".join(b.out_of_scope_topics))
    parts.append("- Do NOT make up information. If you don't know, say so and offer to escalate.")
    parts.append("- Follow the HEAR framework: Hear → Empathize → Act → Resolve.")
    parts.append("- Never blame the customer. Never argue.")

    return "\n".join(parts)


def build_tutor_instruction(profile: TutorProfile) -> str:
    """Build the tutor portion of the system instruction from a profile.

    Deterministic — same profile = same instruction every time.
    """
    s = profile.student
    parts: list[str] = []

    parts.append("TUTOR MODE — ACTIVE")

    if s.student_name:
        parts.append(f"Student: {s.student_name}")
    if s.grade_level:
        parts.append(f"Grade/Level: {s.grade_level}")

    # Subjects
    if s.subjects:
        parts.append("Subjects being studied: " + ", ".join(s.subjects))
    if s.current_courses:
        parts.append("Current courses: " + ", ".join(s.current_courses))

    # Goals
    if s.learning_goals:
        parts.append("Learning goals:")
        for g in s.learning_goals:
            parts.append(f"  - {g}")

    # Strengths & weaknesses
    if s.strengths:
        parts.append("Known strengths: " + ", ".join(s.strengths))
    if s.weaknesses:
        parts.append("Known weaknesses (focus extra help here): " + ", ".join(s.weaknesses))

    # Preferences
    parts.append(f"Learning style: {s.learning_style}")
    parts.append(f"Difficulty level: {s.preferred_difficulty}")

    # Homework mode
    mode_rules = {
        "guide": "Socratic method — NEVER give direct answers. Ask questions that lead to discovery. Help them understand so they can solve it themselves.",
        "explain": "Explain concepts clearly, but still do NOT do the homework for them. They must write their own answers.",
        "review": "Review completed work. Point out errors, explain why they're wrong, suggest corrections. Do not rewrite their work.",
    }
    parts.append(f"Homework help mode: {s.homework_help_mode}")
    parts.append(f"Rule: {mode_rules.get(s.homework_help_mode, mode_rules['guide'])}")

    # Adaptive behavior
    parts.append("")
    parts.append("ADAPTIVE BEHAVIOR:")
    parts.append("- Before teaching, call track_progress(action='summary') to see past performance.")
    parts.append("- If a subject avg score < 50%, lower difficulty and use more analogies.")
    parts.append("- If avg score > 80%, increase difficulty and add challenge problems.")
    parts.append("- After each quiz or exercise, call track_progress(action='record') to log results.")
    parts.append("- Celebrate improvement. Use growth mindset: 'not yet' instead of 'wrong'.")

    # Boundaries
    if s.topics_off_limits:
        parts.append("OFF-LIMITS topics (do not teach): " + ", ".join(s.topics_off_limits))

    parts.append("")
    parts.append("STRICT RULES:")
    parts.append("- Do NOT do homework for the student. Guide them.")
    parts.append("- Do NOT use 'it's easy' or 'obviously' — these shame learners.")
    parts.append("- One concept at a time. Don't overwhelm.")
    parts.append("- If the student is frustrated, take a different approach (analogy, visual, simpler step).")

    return "\n".join(parts)


# ===========================================================================
# Helpers
# ===========================================================================

def _dict_to_customer_care(d: dict) -> CustomerCareProfile:
    """Build CustomerCareProfile from a raw dict, tolerating missing keys."""
    biz_raw = d.get("business", {})
    biz = BusinessProfile(
        business_name=biz_raw.get("business_name", ""),
        tagline=biz_raw.get("tagline", ""),
        industry=biz_raw.get("industry", ""),
        website=biz_raw.get("website", ""),
        products_services=biz_raw.get("products_services", []),
        return_policy=biz_raw.get("return_policy", ""),
        refund_policy=biz_raw.get("refund_policy", ""),
        shipping_policy=biz_raw.get("shipping_policy", ""),
        warranty_policy=biz_raw.get("warranty_policy", ""),
        privacy_note=biz_raw.get("privacy_note", ""),
        business_hours=biz_raw.get("business_hours", ""),
        support_channels=biz_raw.get("support_channels", []),
        escalation_email=biz_raw.get("escalation_email", ""),
        escalation_phone=biz_raw.get("escalation_phone", ""),
        sla_response_time=biz_raw.get("sla_response_time", ""),
        faq=biz_raw.get("faq", []),
        tone=biz_raw.get("tone", "professional and friendly"),
        language=biz_raw.get("language", "English"),
        greeting=biz_raw.get("greeting", ""),
        out_of_scope_topics=biz_raw.get("out_of_scope_topics", []),
        redirect_message=biz_raw.get("redirect_message", ""),
    )
    return CustomerCareProfile(
        version=d.get("version", "1.0"),
        enabled=d.get("enabled", True),
        business=biz,
        tier_rules=d.get("tier_rules", CustomerCareProfile().tier_rules),
    )


def _dict_to_tutor(d: dict) -> TutorProfile:
    """Build TutorProfile from a raw dict, tolerating missing keys."""
    stu_raw = d.get("student", {})
    stu = StudentProfile(
        student_name=stu_raw.get("student_name", ""),
        grade_level=stu_raw.get("grade_level", ""),
        age_group=stu_raw.get("age_group", ""),
        subjects=stu_raw.get("subjects", []),
        current_courses=stu_raw.get("current_courses", []),
        learning_goals=stu_raw.get("learning_goals", []),
        strengths=stu_raw.get("strengths", []),
        weaknesses=stu_raw.get("weaknesses", []),
        learning_style=stu_raw.get("learning_style", "visual"),
        preferred_difficulty=stu_raw.get("preferred_difficulty", "intermediate"),
        language=stu_raw.get("language", "English"),
        topics_off_limits=stu_raw.get("topics_off_limits", []),
        homework_help_mode=stu_raw.get("homework_help_mode", "guide"),
    )
    return TutorProfile(
        version=d.get("version", "1.0"),
        enabled=d.get("enabled", True),
        student=stu,
        socratic_mode=d.get("socratic_mode", True),
    )
