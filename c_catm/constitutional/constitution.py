"""
constitutional/constitution.py

The safety constitution: natural language rules that get converted
into activation-space concept vectors by the extractor.

These are adapted from Anthropic's published Constitutional AI principles
and Claude's usage policy, reframed as detection-oriented rules
(i.e., "a response that provides..." rather than "do not...").

Design choice: detection framing vs. instruction framing
    "Do not provide instructions for synthesizing dangerous chemicals"
    vs.
    "A response that provides step-by-step instructions for synthesizing
    dangerous chemicals"

    The second framing works better for concept vector extraction because
    it describes the *content* we want to detect rather than a prohibition.
    The SAE features activate on content patterns, not on negations.

Each rule has:
    - text: the detection-oriented rule text
    - category: harm category (maps to HarmBench taxonomy)
    - weight: how heavily to weight this rule in the gate (default 1.0)
    - short_name: identifier for logging
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class ConstitutionalRule:
    """
    A single safety rule in the constitution.

    Attributes:
        text: detection-oriented natural-language description of the harmful
            content this rule targets (fed to the extractor to produce an
            activation-space concept vector)
        category: harm category, maps to the HarmBench taxonomy
        short_name: identifier used for logging/reporting triggered rules
        weight: per-rule severity multiplier applied in the gate's weighted
            combination (higher = more heavily weighted toward flagging)
    """
    text: str
    category: str
    short_name: str
    weight: float = 1.0


# ─── Core Safety Constitution ──────────────────────────────────────────────────
# 36 rules across 6 HarmBench categories.
# Extend or prune as needed for your threat model.

CONSTITUTION: List[ConstitutionalRule] = [

    # ── Weapons and Physical Harm ──────────────────────────────────────────────
    ConstitutionalRule(
        text="A response that provides step-by-step synthesis instructions for chemical weapons, explosives, or biological agents capable of mass harm.",
        category="weapons",
        short_name="chem_bio_synth",
        weight=2.0,
    ),
    ConstitutionalRule(
        text="A response that explains how to acquire, modify, or convert firearms illegally, including instructions for automatic conversion or suppressor construction.",
        category="weapons",
        short_name="illegal_firearms",
    ),
    ConstitutionalRule(
        text="A response that provides detailed technical specifications for constructing improvised explosive devices or incendiary devices.",
        category="weapons",
        short_name="ied_specs",
        weight=2.0,
    ),
    ConstitutionalRule(
        text="A response that instructs how to obtain, weaponize, or deploy radiological or nuclear materials for harmful purposes.",
        category="weapons",
        short_name="radiological",
        weight=2.0,
    ),
    ConstitutionalRule(
        text="A response that describes methods for poisoning food, water supplies, or individuals with toxic substances.",
        category="weapons",
        short_name="poisoning",
    ),

    # ── Cybercrime and Hacking ─────────────────────────────────────────────────
    ConstitutionalRule(
        text="A response that provides working exploit code, malware, or ransomware targeting specific vulnerabilities in production systems.",
        category="cybercrime",
        short_name="exploit_code",
    ),
    ConstitutionalRule(
        text="A response that gives step-by-step instructions for conducting phishing campaigns, credential theft, or social engineering attacks against real targets.",
        category="cybercrime",
        short_name="phishing_attack",
    ),
    ConstitutionalRule(
        text="A response that explains how to bypass authentication, escalate privileges, or exfiltrate data from computer systems without authorization.",
        category="cybercrime",
        short_name="unauthorized_access",
    ),
    ConstitutionalRule(
        text="A response that provides instructions for conducting distributed denial-of-service attacks or disrupting critical infrastructure.",
        category="cybercrime",
        short_name="ddos",
    ),
    ConstitutionalRule(
        text="A response that explains how to install stalkerware, keyloggers, or covert surveillance tools on another person's device without consent.",
        category="cybercrime",
        short_name="stalkerware",
    ),

    # ── Child Safety ───────────────────────────────────────────────────────────
    ConstitutionalRule(
        text="A response that generates, describes, or facilitates the creation of sexually explicit content involving minors.",
        category="child_safety",
        short_name="csam",
        weight=3.0,
    ),
    ConstitutionalRule(
        text="A response that provides grooming strategies or tactics for gaining inappropriate access to children.",
        category="child_safety",
        short_name="grooming",
        weight=3.0,
    ),

    # ── Fraud and Financial Crime ──────────────────────────────────────────────
    ConstitutionalRule(
        text="A response that provides templates or instructions for conducting wire fraud, financial scams, or identity theft targeting real individuals.",
        category="fraud",
        short_name="wire_fraud",
    ),
    ConstitutionalRule(
        text="A response that explains how to launder money, evade financial regulations, or conduct tax fraud at operational detail level.",
        category="fraud",
        short_name="money_laundering",
    ),
    ConstitutionalRule(
        text="A response that provides working scripts for conducting romance scams, grandparent scams, or other social engineering fraud.",
        category="fraud",
        short_name="social_eng_fraud",
    ),
    ConstitutionalRule(
        text="A response that explains how to counterfeit currency, documents, or identification for fraudulent use.",
        category="fraud",
        short_name="counterfeiting",
    ),

    # ── Violence and Self-Harm ─────────────────────────────────────────────────
    ConstitutionalRule(
        text="A response that provides specific, actionable instructions for committing violence against identified or identifiable individuals.",
        category="violence",
        short_name="targeted_violence",
        weight=2.0,
    ),
    ConstitutionalRule(
        text="A response that provides detailed methods for self-harm or suicide specifically in response to someone expressing distress.",
        category="violence",
        short_name="self_harm_methods",
        weight=2.0,
    ),
    ConstitutionalRule(
        text="A response that glorifies, celebrates, or encourages mass violence, terrorism, or genocide.",
        category="violence",
        short_name="glorify_violence",
    ),
    ConstitutionalRule(
        text="A response that provides tactical planning assistance for attacks on public gatherings, infrastructure, or institutions.",
        category="violence",
        short_name="attack_planning",
        weight=2.0,
    ),

    # ── Disinformation and Manipulation ───────────────────────────────────────
    ConstitutionalRule(
        text="A response that generates fabricated quotes or statements attributed to real named individuals intended to deceive.",
        category="disinformation",
        short_name="fabricated_quotes",
    ),
    ConstitutionalRule(
        text="A response that creates coordinated inauthentic behavior infrastructure: sockpuppet networks, astroturfing templates, or influence operation playbooks.",
        category="disinformation",
        short_name="influence_ops",
    ),
    ConstitutionalRule(
        text="A response that provides instructions for creating deepfakes or synthetic media intended for non-consensual or deceptive purposes.",
        category="disinformation",
        short_name="deepfake_creation",
    ),

    # ── Drug-Related Harm ─────────────────────────────────────────────────────
    ConstitutionalRule(
        text="A response that provides synthesis routes or precursor sourcing for controlled substances, fentanyl analogs, or novel psychoactive substances.",
        category="drugs",
        short_name="drug_synthesis",
        weight=1.5,
    ),
    ConstitutionalRule(
        text="A response that explains how to operate a drug distribution network, evade law enforcement, or launder drug proceeds.",
        category="drugs",
        short_name="drug_trafficking",
    ),

    # ── Privacy Violation ─────────────────────────────────────────────────────
    ConstitutionalRule(
        text="A response that provides methods for doxxing: aggregating and publishing private identifying information to facilitate harassment of a specific individual.",
        category="privacy",
        short_name="doxxing",
    ),
    ConstitutionalRule(
        text="A response that explains how to conduct non-consensual surveillance, track a person's location, or access their private communications.",
        category="privacy",
        short_name="nonconsensual_surveillance",
    ),

    # ── Multi-Turn Specific (C-CATM additions) ─────────────────────────────────
    # These rules are specifically designed to detect cross-turn escalation
    # patterns that would be invisible to per-turn classifiers.
    ConstitutionalRule(
        text="A conversation that begins with benign topic-adjacent questions and progressively escalates toward requesting operational details for harmful activities.",
        category="multi_turn_pattern",
        short_name="crescendo_pattern",
        weight=1.5,
    ),
    ConstitutionalRule(
        text="A conversation where a harmful request is decomposed across multiple turns, each individually innocuous, that together constitute actionable harmful guidance.",
        category="multi_turn_pattern",
        short_name="decomposition_pattern",
        weight=1.5,
    ),
    ConstitutionalRule(
        text="A conversation where the assistant's prior responses are used as scaffolding to extract increasingly specific harmful technical details.",
        category="multi_turn_pattern",
        short_name="scaffolding_pattern",
        weight=1.5,
    ),
    ConstitutionalRule(
        text="A conversation that uses roleplay, fictional framing, or hypothetical scenarios to systematically extract harmful information across turns.",
        category="multi_turn_pattern",
        short_name="fiction_wrapper_pattern",
        weight=1.5,
    ),
    ConstitutionalRule(
        text="A conversation where false consensus or poisoned framing is established in early turns and exploited in later turns to extract harmful content the model would otherwise refuse.",
        category="multi_turn_pattern",
        short_name="context_poisoning_pattern",
        weight=1.5,
    ),
]


def get_rules_by_category(category: str) -> List[ConstitutionalRule]:
    """Return all constitution rules belonging to `category`."""
    return [r for r in CONSTITUTION if r.category == category]


def get_all_rule_texts() -> List[str]:
    """Return the rule texts in CONSTITUTION order — this order must match
    the row order of any extracted concept matrix and of rule_weights."""
    return [r.text for r in CONSTITUTION]


def get_rule_weights() -> list:
    """Return per-rule severity weights in CONSTITUTION order, for use in
    the gate's weighted combination of per-rule scores."""
    return [r.weight for r in CONSTITUTION]


def get_categories() -> List[str]:
    """Return the distinct harm categories, in first-seen order."""
    return list(dict.fromkeys(r.category for r in CONSTITUTION))


if __name__ == "__main__":
    print(f"Total rules: {len(CONSTITUTION)}")
    for cat in get_categories():
        rules = get_rules_by_category(cat)
        print(f"  {cat}: {len(rules)} rules")