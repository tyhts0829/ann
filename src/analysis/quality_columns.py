"""品質マップで使用する検査項目順序。"""

BASE_COLNAMES = (
    "Foreign_Length_Long",
    "Foreign_Length_Short",
    "Foreign_Size",
    "Lead_Length_L",
    "Lead_Length_R",
    "Lead_Pitch",
    "Work_Xw",
    "Work_Yw",
    "Work_Center_X",
    "Work_Center_Y",
    "Mark_Center_X",
    "Mark_Center_Y",
    "Defect_Length_Long",
    "Defect_Length_Short",
    "Defect_Size",
)

SPEC_ORDER = tuple(
    f"{base_colname}_v{vision_index}"
    for base_colname in BASE_COLNAMES
    for vision_index in (1, 2, 3)
)

CATEGORY_ORDER = ("異物", "リード", "PKGサイズ", "標印", "欠陥")
