# TODO: Consolidate all bias in bios utility stuff
# TODO: Only use strings for keys, only use ints when initializing the dictionary datasets

profession_dict = {
    "accountant": 0,
    "architect": 1,
    "attorney": 2,
    "chiropractor": 3,
    "comedian": 4,
    "composer": 5,
    "dentist": 6,
    "dietitian": 7,
    "dj": 8,
    "filmmaker": 9,
    "interior_designer": 10,
    "journalist": 11,
    "model": 12,
    "nurse": 13,
    "painter": 14,
    "paralegal": 15,
    "pastor": 16,
    "personal_trainer": 17,
    "photographer": 18,
    "physician": 19,
    "poet": 20,
    "professor": 21,
    "psychologist": 22,
    "rapper": 23,
    "software_engineer": 24,
    "surgeon": 25,
    "teacher": 26,
    "yoga_teacher": 27,
}
profession_int_to_str = {v: k for k, v in profession_dict.items()}

gender_dict = {
    "male": 0,
    "female": 1,
}

dataset_metadata = {
    "bias_in_bios": {
        "text_column_name": "hard_text",
        "column1_name": "profession",
        "column2_name": "gender",
        "column1_mapping": profession_dict,
        "column2_mapping": gender_dict,
    }
}
