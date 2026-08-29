# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

"""Generalized specification-variant filtering for ORAN-Lint.

The filter runs after DeBERTa and before contextual LLM verification.  It
recognizes O-RAN pairs that belong to different procedures, components,
directions, configurations, test cases, or explicitly enumerated branches.
Unlike semantic equivalence checks, a variant decision does not require the
remaining text to be identical: different scenarios are expected to have
different behavior and expected results.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


FILTER_VERSION = "generalized-context-aware"
FILTER_OUTPUT_SCHEMA_VERSION = "variant-filter-output"


class FilterVerdict(str, Enum):
    DIFFERENT_VARIANT = "different_variant"
    COUNTER_PAIR = "counter_pair"
    MODAL_DIFFERENCE = "modal_difference"
    NEEDS_DEEP_REVIEW = "needs_deep_review"
    LIKELY_INCONSISTENT = "likely_inconsistent"


class FilterDecision(str, Enum):
    AUTO_NEUTRAL = "auto_neutral"
    SEND_TO_GPT = "send_to_gpt"


@dataclass(frozen=True)
class PatternSpec:
    regex: re.Pattern
    canonical: str


@dataclass(frozen=True)
class AxisDifference:
    axis: str
    side1_values: Tuple[str, ...]
    side2_values: Tuple[str, ...]
    provenance: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ParameterDifference:
    name: str
    side1_values: Tuple[str, ...]
    side2_values: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioContext:
    anchor: str = ""
    anchor_id: Optional[int] = None
    parent_anchor: str = ""
    parent_id: Optional[int] = None
    branch_condition: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class VariantAnalysis:
    verdict: FilterVerdict
    reason: str
    variant_type_1: Optional[str] = None
    variant_type_2: Optional[str] = None
    confidence: float = 0.0
    same_section: bool = False
    same_pdf: bool = False
    decision: FilterDecision = FilterDecision.SEND_TO_GPT
    primary_rule: Optional[str] = None
    matched_rules: List[str] = field(default_factory=list)
    axis_differences: List[AxisDifference] = field(default_factory=list)
    parameter_differences: List[ParameterDifference] = field(default_factory=list)
    scenario_context_1: ScenarioContext = field(default_factory=ScenarioContext)
    scenario_context_2: ScenarioContext = field(default_factory=ScenarioContext)
    scenario_evidence: List[str] = field(default_factory=list)
    hard_vetoes: List[str] = field(default_factory=list)
    detected_axes: Dict[str, List[str]] = field(default_factory=dict)
    evidence: Dict[str, object] = field(default_factory=dict)
    version: str = FILTER_VERSION

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "variant_filter_version": self.version,
            "variant_filter_output_schema_version": FILTER_OUTPUT_SCHEMA_VERSION,
            "variant_filter_verdict": self.verdict.value,
            "variant_filter_decision": self.decision.value,
            "variant_filter_reason": self.reason,
            "variant_filter_confidence": self.confidence,
            "variant_filter_same_section": self.same_section,
            "variant_filter_same_pdf": self.same_pdf,
        }
        if self.decision == FilterDecision.AUTO_NEUTRAL:
            result["variant_filter_proposed_verdict"] = "neutral"
        if self.variant_type_1:
            result["variant_filter_variant_type_1"] = self.variant_type_1
        if self.variant_type_2:
            result["variant_filter_variant_type_2"] = self.variant_type_2
        if self.primary_rule:
            result["variant_filter_primary_rule"] = self.primary_rule
        if self.matched_rules:
            result["variant_filter_matched_rules"] = list(self.matched_rules)
        if self.axis_differences:
            result["variant_filter_axis_differences"] = [
                item.to_dict() for item in self.axis_differences
            ]
        if self.parameter_differences:
            result["variant_filter_parameter_differences"] = [
                item.to_dict() for item in self.parameter_differences
            ]
        scenario_context_1 = self.scenario_context_1.to_dict()
        if any(value not in (None, "") for value in scenario_context_1.values()):
            result["variant_filter_scenario_context_1"] = scenario_context_1
        scenario_context_2 = self.scenario_context_2.to_dict()
        if any(value not in (None, "") for value in scenario_context_2.values()):
            result["variant_filter_scenario_context_2"] = scenario_context_2
        if self.scenario_evidence:
            result["variant_filter_scenario_evidence"] = list(self.scenario_evidence)
        if self.hard_vetoes:
            result["variant_filter_hard_vetoes"] = list(self.hard_vetoes)
        if self.detected_axes:
            result["variant_filter_detected_axes"] = dict(self.detected_axes)
        if self.evidence:
            result["variant_filter_evidence"] = dict(self.evidence)
        return result


def _spec(pattern: str, canonical: str) -> PatternSpec:
    return PatternSpec(re.compile(pattern, re.IGNORECASE), canonical)


# These categories capture recurring intentional O-RAN structural variants.
AXIS_PATTERNS: Mapping[str, Tuple[PatternSpec, ...]] = {
    "operation": (
        _spec(r"\baddition\b|\badd(?:ed|ing)?\b", "ADDITION"),
        _spec(r"\bmodification\b|\bmodify(?:ing|ied)?\b", "MODIFICATION"),
        _spec(r"\bdeletion\b|\bdelete(?:d|s|ing)?\b", "DELETION"),
        _spec(r"\brelease\b|\breleased\b", "RELEASE"),
        _spec(r"\bupdate\b|\bupdated\b", "UPDATE"),
        _spec(r"\bset[-\s]?up\b", "SETUP"),
        _spec(r"\breset\b", "RESET"),
    ),
    "scope": (
        _spec(r"\bintra(?=[-\s])", "INTRA"),
        _spec(r"\binter(?=[-\s])", "INTER"),
    ),
    "direction": (
        _spec(r"\btransmit(?:ted|s|ting)?\b|\bsend(?:s|ing)?\b|\bsent\b", "TX"),
        _spec(r"\breceive(?:d|s|ing)?\b|\breception\b", "RX"),
        _spec(r"\buplink\b|(?<![-\w])UL(?![-\w])", "UL"),
        _spec(r"\bdownlink\b|(?<![-\w])DL(?![-\w])", "DL"),
    ),
    "initiator": (
        _spec(r"\bM-\s*node\b|(?<![-\w])MN(?![-\w])|\bMaster(?:\s+Node)?(?=\s+initiated)", "MASTER"),
        _spec(r"\bS-\s*node\b|(?<![-\w])SN(?![-\w])|\bSecondary(?:\s+Node)?(?=\s+initiated)", "SECONDARY"),
    ),
    "timing": (
        _spec(r"\binitial(?:\s+transmission)?\b|\b(?:in\s+the\s+)?first\s+time\b", "INITIAL"),
        _spec(r"\bretransmission\b", "RETRANSMISSION"),
        _spec(r"\bsubsequent\b", "SUBSEQUENT"),
    ),
    "radio_condition": (
        _spec(r"\bexcellent(?:\s+radio\s+conditions?)?\b", "EXCELLENT"),
        _spec(r"\bgood(?:\s+radio\s+conditions?)?\b", "GOOD"),
        _spec(r"\bfair(?:\s+radio\s+conditions?)?\b", "FAIR"),
        _spec(r"\bpoor(?:\s+radio\s+conditions?)?\b", "POOR"),
    ),
    "component": (
        _spec(r"(?<![A-Za-z0-9])O-CU-CP(?![A-Za-z0-9])", "O-CU-CP"),
        _spec(r"(?<![A-Za-z0-9])O-CU-UP(?![A-Za-z0-9])", "O-CU-UP"),
        _spec(r"(?<![A-Za-z0-9])O-RU(?![A-Za-z0-9-])", "O-RU"),
        _spec(r"(?<![A-Za-z0-9])O-DU(?![A-Za-z0-9-])", "O-DU"),
        _spec(r"(?<![A-Za-z0-9])O-CU(?![A-Za-z0-9-])", "O-CU"),
        _spec(r"(?<![-\w])RIC(?![-\w])", "RIC"),
    ),
    "node_generation": (
        _spec(r"\bMgNB\b", "MgNB"),
        _spec(r"\bSgNB\b", "SgNB"),
        _spec(r"\bgNB-CU\b", "gNB-CU"),
        _spec(r"\bgNB-DU\b", "gNB-DU"),
        _spec(r"\ben-gNB\b", "en-gNB"),
        _spec(r"(?<![-\w])gNB(?![-\w])", "gNB"),
        _spec(r"(?<![-\w])eNB(?![-\w])", "eNB"),
    ),
    "media": (
        _spec(r"\bvoice\b|\bVoLTE\b|\bVoNR\b|\baudio\b", "VOICE"),
        _spec(r"\bvideo\b|\bViLTE\b", "VIDEO"),
    ),
    "duplex": (_spec(r"(?<![-\w])FDD(?![-\w])", "FDD"), _spec(r"(?<![-\w])TDD(?![-\w])", "TDD")),
    "frequency_range": (_spec(r"(?<![-\w])FR1(?![-\w])", "FR1"), _spec(r"(?<![-\w])FR2(?![-\w])", "FR2")),
    "deployment": (_spec(r"(?<![-\w])NSA(?![-\w])", "NSA"), _spec(r"(?<![-\w])SA(?![-\w])", "SA")),
    "test_access": (
        _spec(r"\bnon[-\s]?conducted(?:\s+OTA)?\b|\bOTA\b", "NON_CONDUCTED_OTA"),
        _spec(r"(?<!non[-\s])\bconducted(?:[-\s]+signal)?\b", "CONDUCTED"),
    ),
    "duration": (
        _spec(r"\bundefined[-\s]?(?:duration|period)\b", "UNDEFINED"),
        _spec(r"(?<!un)\bdefined[-\s]?(?:duration|period)\b", "DEFINED"),
    ),
    "format": (
        _spec(r"\bnon[-\s]?static\b|\bdynamic[-\s]?format\b", "DYNAMIC"),
        _spec(r"(?<!non[-\s])\bstatic[-\s]?format\b", "STATIC"),
    ),
    "beam_mode": (
        _spec(r"\bno\s+beamforming\b", "NONE"),
        _spec(r"\bpredefined[-\s]?beam(?:forming)?\b", "PREDEFINED"),
        _spec(r"\bchannel[-\s]?information[-\s]?based\b", "CHANNEL_INFORMATION"),
        _spec(r"\battribute[-\s]?based\b", "ATTRIBUTE"),
        _spec(r"\bweight[-\s]?based\b", "WEIGHT"),
    ),
    "test_polarity": (
        _spec(r"\bpositive\s+(?:test\s+)?(?:case|scenario)\b", "POSITIVE"),
        _spec(r"\bnegative\s+(?:test\s+)?(?:case|scenario)\b", "NEGATIVE"),
    ),
    "message_role": (
        _spec(r"\brequest\b", "REQUEST"),
        _spec(r"\bresponse\b|\breply\b", "RESPONSE"),
        _spec(r"\bnotification\b", "NOTIFICATION"),
    ),
    "api_role": (
        _spec(r"\bAPI\s+Consumer\b|\bconsumer\b|\bclient\b", "CONSUMER"),
        _spec(r"\bAPI\s+Producer\b|\bproducer\b|\bserver\b", "PRODUCER"),
    ),
}


SAME_PDF_AXES = {
    "operation", "scope", "direction", "initiator", "timing", "radio_condition", "media",
    "duplex", "frequency_range", "deployment", "test_access", "duration", "format", "beam_mode",
    "test_polarity", "message_role",
}
SAME_SECTION_AXES = {"component", "node_generation"}
SCENARIO_REQUIRED_AXES = {"api_role"}


KNOWN_PARAMETER_NAMES = (
    "extType", "sectionType", "section type", "symbolMask", "symMask", "rbgMask", "rb", "SymInc",
    "startSymbolAndLength", "numSlots", "numSlotsExt", "slotId", "antMask", "beamId", "Period",
    "SCS", "numerology", "configuration ID", "config ID", "message type", "bit width", "PRB", "RBG",
    "resource block group size",
)
PARAMETER_NAME_RE = "|".join(sorted((re.escape(name) for name in KNOWN_PARAMETER_NAMES), key=len, reverse=True))
PARAMETER_RE = re.compile(
    rf"(?P<name>{PARAMETER_NAME_RE})\s*(?:field\s*)?(?:=|:|\bis\s+set\s+to\b|\bset\s+to\b|\bvalue\s+(?:is|of)\b|\bwith\b)\s*"
    r"(?P<value>0x[0-9a-f]+|0b[01]+|[01]+b|[-+]?\d+(?:\.\d+)?(?:\s*[A-Za-z]+)?|[A-Za-z][A-Za-z0-9_-]*)",
    re.IGNORECASE,
)
XML_PARAMETER_RE = re.compile(r"<(?P<name>[A-Za-z][\w.-]{1,50})>\s*(?P<value>[^<\s][^<]{0,80}?)\s*</(?P=name)>", re.IGNORECASE)

HTTP_METHOD_RE = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\b", re.IGNORECASE)
HTTP_STATUS_RE = re.compile(r"\b(?:1\d\d|2\d\d|3\d\d|4\d\d|5\d\d)\s+(?:OK|Created|Accepted|No Content|Bad Request|Unauthorized|Forbidden|Not Found|Conflict|Error)\b", re.IGNORECASE)
API_ID_RE = re.compile(r"/\{api(?:Consumer|Producer)Id\}(?=/|\b)", re.IGNORECASE)
URI_PATH_RE = re.compile(r"(?:/\{?[A-Za-z][A-Za-z0-9_-]*\}?){1,8}")
FORMULA_RE = re.compile(r"\bformula\b|\blog10\s*\(|=", re.IGNORECASE)
SIGNED_UNIT_RE = re.compile(r"(?<![\w.])[-+]\s*\d+(?:\.\d+)?\s*(?:dB|dBm|Hz|kHz|MHz|GHz|ms|us|ns|%)", re.IGNORECASE)
BOOLEAN_SETTING_RE = re.compile(r"\bwhen\s+set\s+to\s+(?:true|false)\b", re.IGNORECASE)
FORCE_RE = re.compile(r"\bforce(?:s|d)?\b|\bonly\b", re.IGNORECASE)
PREVENT_RE = re.compile(r"\bprevent(?:s|ed)?\b|\bcannot\b|\bnot\b", re.IGNORECASE)
CONFIG_CONTEXT_RE = re.compile(r"\b(?:configur(?:e|ed|ation)|parameter|field|scenario|test\s+case|option|profile|format|mode)\b", re.IGNORECASE)
MALFORMED_TITLE_RE = re.compile(r"<rpc|<notification|message-id=|\bStep\s+\d+\b", re.IGNORECASE)

TEST_ANCHOR_RE = re.compile(
    r"\b(?:Test\s+(?:Case|Scenario)|Scenario|Option)\s*#?\s*"
    r"(?=[A-Za-z0-9.-]*\d)[A-Za-z0-9.-]+"
    r"(?:\s+(?:Requirement|Expected\s+Result))?\b",
    re.IGNORECASE,
)
REQUIREMENT_ANCHOR_RE = re.compile(
    r"\bTest\s+Case\s*#?\s*(?=[A-Za-z0-9.-]*\d)[A-Za-z0-9.-]+\s+"
    r"(?:Requirement|Expected\s+Result)\b",
    re.IGNORECASE,
)
PARENT_ANCHOR_RE = re.compile(
    r"\b(?:following|below)\s+(?:outcomes|cases|branches|scenarios|options)\s+"
    r"(?:are\s+)?(?:considered|defined|supported|listed)\b",
    re.IGNORECASE,
)
BRANCH_CONDITION_PATTERNS = (
    _spec(r"\bfollowing\s+rejection\b|\brejected\s+by\b", "REJECTION_BRANCH"),
    _spec(r"\bfollowing\s+acceptance\b|\baccepted\s+by\b", "ACCEPTANCE_BRANCH"),
    _spec(r"\[\s*IF\s*\]", "IF_BRANCH"),
    _spec(r"\[\s*ELSE\s*\]", "ELSE_BRANCH"),
)

MODAL_PATTERNS = (
    _spec(r"\bshall\b", "SHALL"),
    _spec(r"\bshould\b", "SHOULD"),
    _spec(r"\bmust\b", "MUST"),
    _spec(r"\bmay\b", "MAY"),
)

# These oppositions are only actionable when bounded corpus context proves
# that the statements belong to separate test cases or alternative branches.
# On their own they remain possible inconsistencies and must reach the LLM.
SCENARIO_OUTCOME_PAIRS = (
    (re.compile(r"\b(?:is|are|shall\s+be)\s+(?:properly\s+)?incremented\b", re.IGNORECASE),
     re.compile(r"\b(?:is|are|shall\s+be)\s+not\s+(?:properly\s+)?incremented\b", re.IGNORECASE), "increment_state"),
    (re.compile(r"\b(?:is|are|shall\s+be)\s+enabled\b", re.IGNORECASE),
     re.compile(r"\b(?:is|are|shall\s+be)\s+disabled\b", re.IGNORECASE), "enablement_state"),
    (re.compile(r"\b(?:accept(?:ed|ance)|successful(?:ly)?)\b", re.IGNORECASE),
     re.compile(r"\b(?:reject(?:ed|ion)|fail(?:ed|ure)?)\b", re.IGNORECASE), "result_state"),
    (re.compile(r"\bbefore\b", re.IGNORECASE), re.compile(r"\bafter\b", re.IGNORECASE), "temporal_state"),
)

CRITICAL_ACTOR_SLOT_RE = re.compile(
    r"\b(?:exploited|initiated|triggered|requested|performed)\s+by\s+(?:an?\s+|the\s+)?"
    r"(?P<actor>tenant|host|consumer|producer|client|server)\b",
    re.IGNORECASE,
)


def _normalize(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _reliable_title(title: str) -> bool:
    title = str(title or "").strip()
    return bool(title and len(title) <= 240 and not MALFORMED_TITLE_RE.search(title))


def _extract_values(text: str, specs: Sequence[PatternSpec]) -> Tuple[str, ...]:
    values = {spec.canonical for spec in specs if spec.regex.search(text)}
    return tuple(sorted(values))


def detect_axes(text: str, title: str = "") -> Dict[str, Tuple[str, ...]]:
    source = f"{title}\n{text}" if _reliable_title(title) else text
    return {axis: _extract_values(source, specs) for axis, specs in AXIS_PATTERNS.items()}


def _exclusive_difference(values1: Sequence[str], values2: Sequence[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    only1 = tuple(sorted(set(values1) - set(values2)))
    only2 = tuple(sorted(set(values2) - set(values1)))
    return only1, only2


def _extract_parameters(text: str) -> Dict[str, Tuple[str, ...]]:
    found: Dict[str, set] = defaultdict(set)
    occupied: List[Tuple[int, int]] = []
    for regex in (PARAMETER_RE, XML_PARAMETER_RE):
        for match in regex.finditer(text):
            span = (match.start(), match.end())
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            name = re.sub(r"[^a-z0-9]+", "_", match.group("name").lower()).strip("_")
            value = _normalize(match.group("value"))
            if name and value and len(value) <= 80:
                occupied.append(span)
                found[name].add(value)
    return {name: tuple(sorted(values)) for name, values in found.items()}


def _parameter_differences(text1: str, text2: str) -> List[ParameterDifference]:
    side1, side2 = _extract_parameters(text1), _extract_parameters(text2)
    return [
        ParameterDifference(name, side1[name], side2[name])
        for name in sorted(set(side1) & set(side2))
        if side1[name] != side2[name]
    ]


def extract_branch_condition(text: str) -> str:
    values = _extract_values(text, BRANCH_CONDITION_PATTERNS)
    return values[0] if len(values) == 1 else ""


def _clean_anchor(anchor: str) -> str:
    anchor = re.sub(r"\s+", " ", str(anchor or "")).strip()
    return anchor[:300]


class CorpusContextIndex:
    """Recover bounded test-case and branch context from ordered corpus blocks."""

    def __init__(self, records: Sequence[Mapping[str, object]], max_lookback: int = 12):
        self.records = [dict(record) for record in records]
        self.max_lookback = max_lookback
        self.position_by_id = {record.get("id"): index for index, record in enumerate(self.records)}
        self._context_cache: Dict[object, ScenarioContext] = {}
        self._marker_cache: Dict[int, Tuple[str, str]] = {}

    def _same_scope(self, target: Mapping[str, object], candidate: Mapping[str, object]) -> bool:
        return (
            str(target.get("pdf_file") or "") == str(candidate.get("pdf_file") or "")
            and str(target.get("section_number") or "") == str(candidate.get("section_number") or "")
        )

    def _markers_for(self, index: int) -> Tuple[str, str]:
        if index in self._marker_cache:
            return self._marker_cache[index]
        text = str(self.records[index].get("text") or "")
        anchor_matches = list(REQUIREMENT_ANCHOR_RE.finditer(text)) or list(TEST_ANCHOR_RE.finditer(text))
        parent_matches = list(PARENT_ANCHOR_RE.finditer(text))
        markers = (
            _clean_anchor(anchor_matches[-1].group(0)) if anchor_matches else "",
            _clean_anchor(parent_matches[-1].group(0)) if parent_matches else "",
        )
        self._marker_cache[index] = markers
        return markers

    def context_for(self, record_id: object) -> ScenarioContext:
        if record_id in self._context_cache:
            return self._context_cache[record_id]
        position = self.position_by_id.get(record_id)
        if position is None:
            return ScenarioContext()
        target = self.records[position]
        anchor = ""
        anchor_id: Optional[int] = None
        parent = ""
        parent_id: Optional[int] = None
        lower = max(-1, position - self.max_lookback - 1)
        for index in range(position, lower, -1):
            candidate = self.records[index]
            if not self._same_scope(target, candidate):
                if index != position:
                    break
                continue
            candidate_anchor, candidate_parent = self._markers_for(index)
            if not anchor:
                if candidate_anchor:
                    anchor = candidate_anchor
                    anchor_id = candidate.get("id")
            if not parent:
                if candidate_parent:
                    parent = candidate_parent
                    parent_id = candidate.get("id")
            if anchor and parent:
                break
        own_text = str(target.get("text") or "")
        context = ScenarioContext(anchor, anchor_id, parent, parent_id, extract_branch_condition(own_text))
        self._context_cache[record_id] = context
        return context


def _scenario_evidence(
    context1: ScenarioContext,
    context2: ScenarioContext,
    same_pdf: bool,
    same_section: bool,
) -> List[str]:
    evidence: List[str] = []
    if same_pdf and same_section and context1.anchor and context2.anchor:
        if _normalize(context1.anchor) != _normalize(context2.anchor):
            evidence.append("distinct_local_scenario_anchors")
    if (
        same_pdf
        and same_section
        and context1.branch_condition
        and context2.branch_condition
        and context1.branch_condition != context2.branch_condition
        and context1.parent_anchor
        and context2.parent_anchor
        and _normalize(context1.parent_anchor) == _normalize(context2.parent_anchor)
    ):
        evidence.append("alternative_branches_under_common_parent")
    return evidence


def _provenance(axis: str, same_pdf: bool, same_section: bool, has_scenario: bool) -> Optional[str]:
    if axis in SAME_SECTION_AXES:
        return "same_pdf_same_section" if same_pdf and same_section else None
    if axis in SAME_PDF_AXES:
        return "same_pdf" if same_pdf else None
    if axis in SCENARIO_REQUIRED_AXES:
        return "explicit_scenario_context" if same_pdf and has_scenario else None
    return None


def _component_incomplete(text1: str, text2: str, axes1: Mapping[str, Sequence[str]], axes2: Mapping[str, Sequence[str]]) -> bool:
    for axis in ("component", "node_generation"):
        values1, values2 = set(axes1.get(axis, ())), set(axes2.get(axis, ()))
        if values1 and values2 and (values1 < values2 or values2 < values1):
            return True
    return False


def _formula_sign_difference(text1: str, text2: str) -> bool:
    if not (FORMULA_RE.search(text1) and FORMULA_RE.search(text2)):
        return False
    signs1 = tuple(re.sub(r"\s+", "", match.group(0).lower()) for match in SIGNED_UNIT_RE.finditer(text1))
    signs2 = tuple(re.sub(r"\s+", "", match.group(0).lower()) for match in SIGNED_UNIT_RE.finditer(text2))
    return bool(signs1 and signs2 and signs1 != signs2)


def _boolean_opposite_effect(text1: str, text2: str) -> bool:
    if not (BOOLEAN_SETTING_RE.search(text1) and BOOLEAN_SETTING_RE.search(text2)):
        return False
    return (bool(FORCE_RE.search(text1)), bool(PREVENT_RE.search(text1))) != (
        bool(FORCE_RE.search(text2)), bool(PREVENT_RE.search(text2))
    )


def _http_signature(regex: re.Pattern, text: str) -> Tuple[str, ...]:
    return tuple(sorted({match.group(0).upper() for match in regex.finditer(text)}))


def _api_path_variant(text1: str, text2: str) -> bool:
    has_id_1, has_id_2 = bool(API_ID_RE.search(text1)), bool(API_ID_RE.search(text2))
    if has_id_1 == has_id_2:
        return False

    def normalized_paths(text: str) -> set:
        return {
            API_ID_RE.sub("", match.group(0)).rstrip("/").lower()
            for match in URI_PATH_RE.finditer(text)
        }

    return bool(normalized_paths(text1) & normalized_paths(text2))


def _scenario_outcome_differences(text1: str, text2: str) -> List[str]:
    differences: List[str] = []
    for positive, negative, label in SCENARIO_OUTCOME_PAIRS:
        side1 = (bool(positive.search(text1)), bool(negative.search(text1)))
        side2 = (bool(positive.search(text2)), bool(negative.search(text2)))
        if side1 in ((True, False), (False, True)) and side2 in ((True, False), (False, True)) and side1 != side2:
            differences.append(label)
        elif side1 == (True, True) and side2 == (True, True):
            differences.append(f"{label}_allocation")
    return differences


def _critical_actor_difference(text1: str, text2: str) -> bool:
    actors1 = {match.group("actor").upper() for match in CRITICAL_ACTOR_SLOT_RE.finditer(text1)}
    actors2 = {match.group("actor").upper() for match in CRITICAL_ACTOR_SLOT_RE.finditer(text2)}
    only1, only2 = actors1 - actors2, actors2 - actors1
    return bool(only1 and only2)


def analyze_pair(
    text1: str,
    text2: str,
    section1: str = "",
    section2: str = "",
    pdf1: str = "",
    pdf2: str = "",
    *,
    title1: str = "",
    title2: str = "",
    context1: Optional[ScenarioContext] = None,
    context2: Optional[ScenarioContext] = None,
    bge_cos: Optional[float] = None,
    tfidf_cos: Optional[float] = None,
) -> VariantAnalysis:
    """Classify whether a candidate pair represents intentional variants."""
    text1, text2 = str(text1 or ""), str(text2 or "")
    same_pdf = bool(pdf1 and pdf2 and str(pdf1) == str(pdf2))
    same_section = bool(same_pdf and section1 and section2 and str(section1) == str(section2))
    context1 = context1 or ScenarioContext(branch_condition=extract_branch_condition(text1))
    context2 = context2 or ScenarioContext(branch_condition=extract_branch_condition(text2))
    scenario_evidence = _scenario_evidence(context1, context2, same_pdf, same_section)
    has_scenario = bool(scenario_evidence)

    axes1, axes2 = detect_axes(text1, title1), detect_axes(text2, title2)
    axis_differences: List[AxisDifference] = []
    raw_axis_differences: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...]]] = {}
    detected_axes: Dict[str, List[str]] = {}
    for axis in AXIS_PATTERNS:
        only1, only2 = _exclusive_difference(axes1[axis], axes2[axis])
        if only1 and only2:
            raw_axis_differences[axis] = (only1, only2)
            detected_axes[axis] = ["|".join(only1), "|".join(only2)]
            provenance = _provenance(axis, same_pdf, same_section, has_scenario)
            if provenance:
                axis_differences.append(AxisDifference(axis, only1, only2, provenance))

    parameter_differences = _parameter_differences(text1, text2)
    eligible_parameters = bool(
        parameter_differences
        and same_pdf
        and same_section
        and (has_scenario or bool(axis_differences))
        and CONFIG_CONTEXT_RE.search(text1)
        and CONFIG_CONTEXT_RE.search(text2)
    )

    matched_rules = [f"axis:{item.axis}" for item in axis_differences]
    matched_rules.extend(f"parameter:{item.name}" for item in parameter_differences if eligible_parameters)
    scenario_outcomes = _scenario_outcome_differences(text1, text2) if has_scenario else []
    matched_rules.extend(f"scenario_outcome:{item}" for item in scenario_outcomes)
    if "alternative_branches_under_common_parent" in scenario_evidence:
        matched_rules.append("scenario_branch:alternative_under_common_parent")

    both_counters = all(
        re.search(r"\b(?:counter|measurement|metric|KPI|incremented|decremented)s?\b", text, re.IGNORECASE)
        for text in (text1, text2)
    )
    direction_diff = raw_axis_differences.get("direction")
    if both_counters and direction_diff:
        if not any(item.axis == "direction" for item in axis_differences):
            axis_differences.append(AxisDifference("direction", direction_diff[0], direction_diff[1], "counter_pair"))
        matched_rules.append("counter_direction")

    api_variant = _api_path_variant(text1, text2)
    if api_variant and _http_signature(HTTP_METHOD_RE, text1) == _http_signature(HTTP_METHOD_RE, text2):
        matched_rules.append("api_path_optional_identifier")

    hard_vetoes: List[str] = []
    incomplete_component = _component_incomplete(text1, text2, axes1, axes2)
    high_similarity = bge_cos is None or float(bge_cos) >= 0.97
    if incomplete_component and same_section and high_similarity:
        hard_vetoes.append("incomplete_component_substitution")
    if _critical_actor_difference(text1, text2) and not has_scenario:
        hard_vetoes.append("unresolved_actor_role_difference")
    if _formula_sign_difference(text1, text2):
        hard_vetoes.append("formula_sign_or_operator_difference")
    if _boolean_opposite_effect(text1, text2):
        hard_vetoes.append("same_boolean_setting_opposite_effect")
    methods1, methods2 = _http_signature(HTTP_METHOD_RE, text1), _http_signature(HTTP_METHOD_RE, text2)
    operation_variant = any(item.axis == "operation" for item in axis_differences)
    polarity_variant = any(item.axis == "test_polarity" for item in axis_differences)
    if methods1 != methods2 and not operation_variant:
        hard_vetoes.append("http_method_difference")
    statuses1, statuses2 = _http_signature(HTTP_STATUS_RE, text1), _http_signature(HTTP_STATUS_RE, text2)
    if statuses1 != statuses2 and not (polarity_variant or has_scenario):
        hard_vetoes.append("http_status_difference")
    if parameter_differences and not eligible_parameters and not axis_differences:
        hard_vetoes.append("same_scenario_parameter_value_difference")

    matched_rules = sorted(set(matched_rules))
    hard_vetoes = sorted(set(hard_vetoes))
    evidence = {
        "bge_cos": float(bge_cos) if bge_cos is not None else None,
        "tfidf_cos": float(tfidf_cos) if tfidf_cos is not None else None,
        "same_pdf": same_pdf,
        "same_section": same_section,
        "scenario_proven": has_scenario,
        "parameter_bundle_eligible": eligible_parameters,
        "text1_sha256": hashlib.sha256(text1.encode("utf-8")).hexdigest(),
        "text2_sha256": hashlib.sha256(text2.encode("utf-8")).hexdigest(),
    }

    if matched_rules and not hard_vetoes:
        if scenario_evidence:
            confidence = 0.95
        elif any(item.provenance == "same_pdf_same_section" for item in axis_differences):
            confidence = 0.93
        elif axis_differences:
            confidence = 0.88
        else:
            confidence = 0.90
        primary = matched_rules[0]
        verdict = FilterVerdict.COUNTER_PAIR if "counter_direction" in matched_rules else FilterVerdict.DIFFERENT_VARIANT
        values1 = ", ".join("/".join(item.side1_values) for item in axis_differences) or context1.anchor or context1.branch_condition
        values2 = ", ".join("/".join(item.side2_values) for item in axis_differences) or context2.anchor or context2.branch_condition
        return VariantAnalysis(
            verdict=verdict,
            decision=FilterDecision.AUTO_NEUTRAL,
            reason=f"Distinct intentional variant evidence: {', '.join(matched_rules)}.",
            variant_type_1=values1 or None,
            variant_type_2=values2 or None,
            confidence=confidence,
            same_section=same_section,
            same_pdf=same_pdf,
            primary_rule=primary,
            matched_rules=matched_rules,
            axis_differences=axis_differences,
            parameter_differences=parameter_differences,
            scenario_context_1=context1,
            scenario_context_2=context2,
            scenario_evidence=scenario_evidence,
            hard_vetoes=[],
            detected_axes=detected_axes,
            evidence=evidence,
        )

    modal1 = _extract_values(text1, MODAL_PATTERNS)
    modal2 = _extract_values(text2, MODAL_PATTERNS)
    verdict = FilterVerdict.MODAL_DIFFERENCE if modal1 != modal2 and modal1 and modal2 else FilterVerdict.NEEDS_DEEP_REVIEW
    reason = "Variant evidence was vetoed by a same-scenario conflict safeguard." if hard_vetoes else "No eligible generalized variant evidence was found."
    return VariantAnalysis(
        verdict=verdict,
        decision=FilterDecision.SEND_TO_GPT,
        reason=reason,
        confidence=0.30,
        same_section=same_section,
        same_pdf=same_pdf,
        primary_rule=matched_rules[0] if matched_rules else None,
        matched_rules=matched_rules,
        axis_differences=axis_differences,
        parameter_differences=parameter_differences,
        scenario_context_1=context1,
        scenario_context_2=context2,
        scenario_evidence=scenario_evidence,
        hard_vetoes=hard_vetoes,
        detected_axes=detected_axes,
        evidence=evidence,
    )


def should_skip_gpt(analysis: VariantAnalysis, confidence_threshold: float = 0.70) -> bool:
    return analysis.decision == FilterDecision.AUTO_NEUTRAL and analysis.confidence >= confidence_threshold


def get_enhanced_prompt_context(analysis: VariantAnalysis) -> str:
    parts: List[str] = []
    if analysis.scenario_evidence:
        parts.append("Recovered scenario evidence: " + ", ".join(analysis.scenario_evidence) + ".")
    if analysis.matched_rules:
        parts.append("Detected variant rules: " + ", ".join(analysis.matched_rules) + ".")
    if analysis.hard_vetoes:
        parts.append("Potential same-scenario conflict signals: " + ", ".join(analysis.hard_vetoes) + ".")
    if analysis.same_section and analysis.same_pdf:
        parts.append("Both statements occur in the same section; check whether they belong to separate sibling test cases.")
    return " ".join(parts)


def annotate_pair(row: Mapping[str, object], context_index: Optional[CorpusContextIndex] = None) -> Dict[str, object]:
    context1 = context_index.context_for(row.get("id1")) if context_index else None
    context2 = context_index.context_for(row.get("id2")) if context_index else None
    analysis = analyze_pair(
        str(row.get("text1") or ""),
        str(row.get("text2") or ""),
        str(row.get("id1_section_number") or ""),
        str(row.get("id2_section_number") or ""),
        str(row.get("id1_pdf_file") or ""),
        str(row.get("id2_pdf_file") or ""),
        title1=str(row.get("id1_section_title") or ""),
        title2=str(row.get("id2_section_title") or ""),
        context1=context1,
        context2=context2,
        bge_cos=row.get("bge_cos"),
        tfidf_cos=row.get("tfidf_cos"),
    )
    annotated = dict(row)
    annotated.update(analysis.to_dict())
    return annotated


def filter_pairs(
    pairs: Sequence[Mapping[str, object]], context_index: Optional[CorpusContextIndex] = None
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    automatic: List[Dict[str, object]] = []
    review: List[Dict[str, object]] = []
    for pair in pairs:
        annotated = annotate_pair(pair, context_index)
        if annotated["variant_filter_decision"] == FilterDecision.AUTO_NEUTRAL.value:
            annotated["auto_verdict"] = "neutral"
            automatic.append(annotated)
        else:
            review.append(annotated)
    return automatic, review
