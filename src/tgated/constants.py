ORGAN_TEXT_PROMPTS = {
    "liver": "liver",
    "spleen": "spleen",
    "pancreas": "pancreas",
    "gallbladder": "gallbladder",
    "esophagus": "esophagus",
    "duodenum": "duodenum",
    "stomach": "stomach",
    "aorta": "aorta",
    "r_kidney": "right kidney",
    "l_kidney": "left kidney",
    "ivc": "inferior vena cava",
    "r_adrenal": "right adrenal gland",
    "l_adrenal": "left adrenal gland",
}

ORGAN_WEIGHTS = {
    "r_adrenal": 8.0,
    "l_adrenal": 8.0,
    "esophagus": 8.0,
    "duodenum": 5.0,
    "pancreas": 3.0,
    "stomach": 2.0,
    "aorta": 1.0,
    "ivc": 1.0,
    "gallbladder": 2.0,
    "r_kidney": 1.0,
    "l_kidney": 1.0,
    "spleen": 1.0,
    "liver": 1.0,
}

LATERALIZED_ORGANS = {"r_kidney", "l_kidney", "r_adrenal", "l_adrenal"}
NEGATIVE_RATIO = 0.10
