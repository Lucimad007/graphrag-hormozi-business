from enum import StrEnum

ONTOLOGY_VERSION = "1.0.0"


class EntityType(StrEnum):
    BUSINESS = "Business"
    CUSTOMER = "Customer"
    OFFER = "Offer"
    PROBLEM = "Problem"
    METRIC = "Metric"
    STRATEGY = "Strategy"
    TACTIC = "Tactic"
    CONCEPT = "Concept"
    CONSTRAINT = "Constraint"
    PREREQUISITE = "Prerequisite"
    OUTCOME = "Outcome"
    LEAD_SOURCE = "LeadSource"
    BUSINESS_STAGE = "BusinessStage"
    EVIDENCE = "Evidence"
    SOURCE = "Source"


class RelationType(StrEnum):
    SOLVES = "SOLVES"
    IMPROVES = "IMPROVES"
    REQUIRES = "REQUIRES"
    TARGETS = "TARGETS"
    CAUSES = "CAUSES"
    RELATED_TO = "RELATED_TO"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    DERIVED_FROM = "DERIVED_FROM"
    APPLIES_TO = "APPLIES_TO"
    MEASURED_BY = "MEASURED_BY"
    PART_OF = "PART_OF"
