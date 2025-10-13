class PlaygroundTools:
    WEB: str = "web"


class ConversationType:
    CHAT: str = "chat"


class ContextType:
    VALIDATION_LEGAL: str = "validation_legal"
    VALIDATION_TECHNICAL: str = "validation_technical"
    TABLE_OF_CONTENT: str = "table_of_content"
    DEEP_RESEARCH: str = "deep_research"
    USER_PREFERENCE: str = "user_preference"
    CONTENT: str = "long_term_memory"
    COST_INFRASTRUCTURE: str = "rfp_infrastructure"
    COST_LICENSE: str = "rfp_license"
    COST_HR: str = "hourly_wages"
    RFP_TEXT: str = "rfp_text"


BATCH_CONTEXT_PATHS = {
    ContextType.VALIDATION_LEGAL: "stage/validation/validation_results.json",
    ContextType.VALIDATION_TECHNICAL: "stage/validation/validation_results.json",
    ContextType.TABLE_OF_CONTENT: "stage/pre_steps/table_of_content.json",
    ContextType.DEEP_RESEARCH: "stage/deep_research/deep_research.json",
    ContextType.USER_PREFERENCE: "stage/user_preferences/user_preferences.json",
    ContextType.CONTENT: "content/long_term_memory.json",
    ContextType.COST_INFRASTRUCTURE: "stage/costing/rfp_infrastructure.json",
    ContextType.COST_LICENSE: "stage/costing/rfp_license.json",
    ContextType.COST_HR: "stage/costing/hourly_wages.json",
    ContextType.RFP_TEXT: "rfp_text.json",
    "rfp_text": "rfp_text.json",
    "user_preferences": "stage/user_preferences/user_preferences.json",
}
