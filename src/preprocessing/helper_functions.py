import os
import pickle
import re

import numpy as np
import pandas as pd
import yaml


def load_data(config_path, nrows=None):
    """
    This function loads the data and does some preliminary preprocessing that makes the
    following code more readable (mainly renaming and preselecting columns). Because this is a lot
    of manual processing, this function is not very elegant, but I do not have the time to refactor
    this yet.
    What it mainly does is
        - rename certain variables to have consistent naming conventions (e.g., bfi2_s_1) across variables
            and across samples (if questionnaires are identical across ESM-samples, names should also be identical)
        - filter the states and traits including only relevant variables specified in the config
        - stores the processed state and trait dfs with the variables needed in a Dict

    Args:
        config_path: str, relative path to YAML config
        nrows: int, number of rows to load (limited rows were used for certain tests)

    Returns:
        result_dct: Dict containing the preliminary preprocessed trait and state df of the ESM-samples
            specified in the config
    """
    with open(config_path, "r") as f:
        config_raw = yaml.safe_load(f)
    config = config_raw["general"]
    filter_traits_cfg = config_raw["trait_data"]
    filter_states_cfg = config_raw["state_data"]
    result_dct = dict()

    # get raw data path from config
    raw_data_path = config["load_data"]["data_path"]

    # The following Dict includes all person-level features we aim to keep for the analysis for each ESM-sample
    tmp_dct_traits = dict()
    for sample in config["samples_for_analysis"]:
        tmp_dct_traits[f"{sample}_dem_lst"] = list(
            dict.fromkeys(
                [
                    f"{d['name']}_{tp}"  # simplify?
                    for d in filter_traits_cfg["socio_demographics"]
                    for _sample_name in d["time_of_assessment"]
                    if sample in d["time_of_assessment"]
                    for tp in d["time_of_assessment"][sample]
                ]
            )
        )
        tmp_dct_traits[f"{sample}_pers_lst"] = list(
            dict.fromkeys(
                [
                    f"{d['name']}_{item_nr}_{tp}"
                    for d in filter_traits_cfg["personality"]
                    for _sample_name in d["time_of_assessment"]
                    if sample in d["time_of_assessment"]
                    for tp in d["time_of_assessment"][sample]
                    for item_nr in range(1, d["number_of_items"][sample] + 1)
                ]
            )
        )
        tmp_dct_traits[f"{sample}_pol_soc_lst"] = list(
            dict.fromkeys(
                [
                    f"{d['name']}_{item_nr}_{tp}"
                    for d in filter_traits_cfg["polit_soc_attitudes"]
                    for _sample_name in d["time_of_assessment"]
                    if sample in d["time_of_assessment"]
                    for tp in d["time_of_assessment"][sample]
                    for item_nr in range(1, d["number_of_items"][sample] + 1)
                ]
            )
        )
        tmp_dct_traits[f"{sample}_trait_lst"] = (
            tmp_dct_traits[f"{sample}_dem_lst"]
            + tmp_dct_traits[f"{sample}_pers_lst"]
            + tmp_dct_traits[f"{sample}_pol_soc_lst"]
            + config["trait_columns_to_keep"]
        )

        # add FLAG columns to Trait "columns to keep" if we exclude flagged samples
        if config["exclude_flagged_data"][sample]["traits"]:
            tmp_dct_traits[f"{sample}_trait_lst"].extend(
                [
                    flag_col
                    for flag_col in config["exclude_flagged_data"][sample]["traits"]
                ]
            )

    # The following Dict includes all variables we need for the state analysis (i.e., extraction of reactivities)
    tmp_dct_states = dict()
    for sample in config["samples_for_analysis"]:
        tmp_dct_states[f"{sample}_state_lst"] = list()
        try:
            tmp_dct_states[f"{sample}_state_lst"].append(
                [
                    soc_int_var
                    for soc_int_var, var_info in filter_states_cfg["ssc"][
                        "social_interaction_vars"
                    ].items()
                    if sample in var_info["samples"]
                ]
                + filter_states_cfg["general"]["well_being_items"][sample][
                    "positive_affect"
                ]
                + filter_states_cfg["general"]["well_being_items"][sample][
                    "negative_affect"
                ]
                + config["state_columns_to_keep"]
            )
        except KeyError:
            tmp_dct_states[f"{sample}_state_lst"].append(
                filter_states_cfg["general"]["well_being_items"][sample][
                    "positive_affect"
                ]
                + filter_states_cfg["general"]["well_being_items"][sample][
                    "negative_affect"
                ]
                + config["state_columns_to_keep"]
            )

        if config["exclude_flagged_data"][sample]["states"]:
            tmp_dct_states[f"{sample}_state_lst"][0].extend(
                [
                    flag_col
                    for flag_col in config["exclude_flagged_data"][sample]["states"]
                ]
            )

    # At least one ESM-sample has to be specified
    if len(config["samples_for_analysis"]) == 0:
        raise ValueError("No ESM sample specified")

    ### Specific preprocessing for CoCo International
    if "coco_int" in config["samples_for_analysis"]:
        coco_int_path = os.path.join(raw_data_path, 'coco_int')

        df_coco_int_states = pd.read_csv(
            os.path.join(coco_int_path, "CoCo_states.csv"),
            delimiter=",", nrows=nrows
        )
        df_coco_int_traits = pd.read_csv(
            os.path.join(coco_int_path, "CoCo_traits.csv"),
            delimiter=",",
            encoding="latin",
            nrows=nrows,
        )

        # Rename inverse coded columns for simplicity (an r in the column name means that this item was already recoded)
        # These columns from CoCo International are excluded from inverse coding in the BasePreprocessor class
        pattern = re.compile(r"(\w*)(\d+)(r_)(\w*)")
        column_mappings = {
            col: pattern.sub(r"\1\2_\4", col)
            for col in df_coco_int_traits.columns
            if pattern.match(col)
        }
        df_coco_int_traits.rename(columns=column_mappings, inplace=True)

        # rename id column -> consistent with other samples
        df_coco_int_traits.rename(columns={"participant": "id"}, inplace=True)
        df_coco_int_states.rename(columns={"participant": "id"}, inplace=True)

        # rename columns of stab and plas to remain sequential logic of item numbers
        for tp in ["t1", "t2"]:
            count_stab = 1
            for num in [1, 3, 8, 10, 17]:
                df_coco_int_traits.rename(
                    columns={f"stab_{num}_{tp}": f"stab_{count_stab}_{tp}"},
                    inplace=True,
                )
                count_stab += 1
        for tp in ["t1", "t2"]:
            count_plas = 1
            for num in [1, 3, 6, 11, 15]:
                df_coco_int_traits.rename(
                    columns={f"plas_{num}_{tp}": f"plas_{count_plas}_{tp}"},
                    inplace=True,
                )
                count_plas += 1

        # Rename loneliness to uls -> consistent with other samples
        pattern = re.compile(r"(loneliness)(_\d+_\w*)")
        column_mappings = {
            col: pattern.sub(r"uls\2", col)
            for col in df_coco_int_traits.columns
            if pattern.match(col)
        }
        df_coco_int_traits.rename(columns=column_mappings, inplace=True)

        # Rename narq to narqs -> consistent with other samples
        pattern = re.compile(r"(narq)(_\d+_\w*)")
        column_mappings = {
            col: pattern.sub(r"\1s\2", col)
            for col in df_coco_int_traits.columns
            if pattern.match(col)
        }
        df_coco_int_traits.rename(columns=column_mappings, inplace=True)

        # Rename certain other columns that don't follow the logic {ques}_{nr}_tp
        for tp in ["t1", "t2"]:
            df_coco_int_traits.rename(
                columns={
                    f"self_esteem_{tp}": f"self_esteem_1_{tp}",
                    f"trust_general_{tp}": f"pol_trust_1_{tp}",
                    f"trust_government_{tp}": f"pol_trust_2_{tp}",
                    f"trust_science_{tp}": f"pol_trust_3_{tp}",
                    f"satisfaction_democracy_{tp}": f"pol_att_other_1_{tp}",
                    f"political_left_right_{tp}": f"pol_att_other_2_{tp}",
                    f"outgroup_quantity_{tp}": f"pol_outgroup_1_{tp}",
                    f"outgroup_quality_{tp}": f"pol_outgroup_2_{tp}",
                    f"religiosity_{tp}": f"pol_att_other_3_{tp}",
                    f"spirituality_{tp}": f"pol_att_other_4_{tp}",
                    f"subjective_status_{tp}": f"pol_att_other_5_{tp}",
                    f"economic_marginalization1_{tp}": f"wgm_1_{tp}",
                    f"political_marginalization1_{tp}": f"wgm_2_{tp}",
                    f"cultural_marginalization1_{tp}": f"wgm_3_{tp}",
                    f"economic_marginalization2_{tp}": f"wgm_4_{tp}",
                    f"political_marginalization2_{tp}": f"wgm_5_{tp}",
                    f"cultural_marginalization2_{tp}": f"wgm_6_{tp}",
                },
                inplace=True,
            )
        # political efficacy
        for tp in ["t1", "t2"]:
            for num in [1, 2, 3, 4]:
                df_coco_int_traits.rename(
                    columns={
                        f"political_efficacy{num}_{tp}": f"political_efficacy_{num}_{tp}"
                    },
                    inplace=True,
                )
        # threat perception
        for tp in ["t1", "t2"]:
            for num in [1, 2, 3, 4, 5, 6]:
                df_coco_int_traits.rename(
                    columns={
                        f"threat_perception{num}_{tp}": f"threat_perception_{num}_{tp}"
                    },
                    inplace=True,
                )
        # media
        for tp in ["t1", "t2"]:
            for num in [1, 2]:
                df_coco_int_traits.rename(
                    columns={f"media{num}_{tp}": f"media_{num}_{tp}"}, inplace=True
                )

        # State processing
        # align column names that track the esm tp
        df_coco_int_states.rename(
            columns={"created_individual": "created_esm_timepoint"}, inplace=True
        )

        # Preprocess column that assess the interaction context (face-to-face / computer-mediated)
        # 0 is cmc, 1 is ftf, if it is mixed (cmc + ftf) assign 1
        df_coco_int_states["interaction_medium_binary"] = df_coco_int_states[
            "selection_medium"
        ].apply(lambda x: 1 if isinstance(x, str) and "1" in x else 0)
        tmp_dct_states["coco_int_state_lst"][0].append("interaction_medium_binary")

        # Clean Dct with state and trait DF for Coco Int with all variables needed
        result_dct["coco_int"] = [
            df_coco_int_traits[tmp_dct_traits["coco_int_trait_lst"]],
            df_coco_int_states[tmp_dct_states["coco_int_state_lst"][0]],
        ]

    ### Specific Preprocessing for Emotions
    if "emotions" in config["samples_for_analysis"]:
        emotions_path = os.path.join(raw_data_path, 'emo_s2')
        df_emotions_states = pd.read_csv(
            os.path.join(emotions_path, "emotions_states.csv"), delimiter=",", nrows=nrows
        )
        df_emotions_traits_raw = pd.read_csv(
            os.path.join(emotions_path, "emotions_traits.csv"), delimiter=",", nrows=nrows
        )

        # Note: Trait DF IDs are flawed, use Traits from State DF
        df_emotions_traits = df_emotions_states[
            df_emotions_traits_raw.columns.intersection(df_emotions_states.columns)
        ]
        df_emotions_traits = df_emotions_traits.drop_duplicates(subset="id")
        df_emotions_traits = df_emotions_traits.reset_index(drop=True)

        # Create a t1 suffix for column names that are missing it for aligned naming conventions
        pattern = re.compile(
            r"^(?:(?!_t[1234]$).)*$"
        )  # matches all columns not followed by t_2, t_3 or t_4
        column_mappings = {
            column_name: f"{column_name}_t1"
            for column_name in df_emotions_traits.columns
            if column_name != "id" and re.match(pattern, column_name)
        }
        df_emotions_traits = df_emotions_traits.rename(columns=column_mappings)

        # rename demographics (consistent with coco int + and coco_ut)
        for tp in ["t1", "t2", "t3"]:
            df_emotions_traits.rename(
                columns={
                    f"gender_{tp}": f"sex_{tp}",
                    f"educational_status_{tp}": f"educational_attainment_{tp}",
                    f"occupational_status_{tp}": f"professional_status_{tp}",
                    f"household_{tp}": f"quantity_household_{tp}",
                    f"parents_{tp}": f"int_part_parents_{tp}",
                    f"grandparents_{tp}": f"int_part_grandparents_{tp}",
                    f"siblings_{tp}": f"int_part_siblings_{tp}",
                    f"children_{tp}": f"int_part_children_{tp}",
                    f"partner_{tp}": f"int_part_partner_{tp}",
                },
                inplace=True,
            )
        # rename one-item scale according to the common naming conventions
        for tp in ["t1", "t2", "t3", "t4"]:
            df_emotions_traits.rename(
                columns={
                    "rses_1_" + tp: "self_esteem_1_" + tp,
                    "political_orientation_" + tp: "political_orientation_1_" + tp,
                },
                inplace=True,
            )

        # merge and rename state columns assessing well-being, consistent with other samples
        df_emotions_states["state_success"] = df_emotions_states[
            "int_success"
        ].combine_first(df_emotions_states["occup_success"])
        df_emotions_states["state_relaxed"] = df_emotions_states[
            "int_relaxed"
        ].combine_first(df_emotions_states["occup_relaxed"])
        df_emotions_states["state_proud"] = df_emotions_states[
            "int_proud"
        ].combine_first(df_emotions_states["occup_proud"])
        df_emotions_states["state_enthusiastic"] = df_emotions_states[
            "int_enthusiastic"
        ].combine_first(df_emotions_states["occup_enthusiastic"])

        df_emotions_states["state_angry"] = df_emotions_states[
            "int_angry"
        ].combine_first(df_emotions_states["occup_angry"])
        df_emotions_states["state_sad"] = df_emotions_states["int_sad"].combine_first(
            df_emotions_states["occup_sad"]
        )
        df_emotions_states["state_anxious"] = df_emotions_states[
            "int_anxious"
        ].combine_first(df_emotions_states["occup_anxious"])
        df_emotions_states["state_lonely"] = df_emotions_states[
            "int_lonely"
        ].combine_first(df_emotions_states["occup_lonely"])

        # rename social situation variables to be consistent with CoCo International and CoCo UT
        df_emotions_states.rename(
            columns={
                f"interaction": f"social_interaction",
                f"no_interactionpartner": f"interaction_quantity",
            },
            inplace=True,
        )
        df_emotions_states.rename(
            columns={"created_esm": "created_esm_timepoint"}, inplace=True
        )

        # rename pa / na state variables
        df_emotions_states.rename(
            columns={
                "state_relaxed": "state_pa1",
                "state_proud": "state_pa2",
                "state_enthusiastic": "state_pa3",
                "state_success": "state_pa4",
                "state_angry": "state_na1",
                "state_sad": "state_na2",
                "state_anxious": "state_na3",
                "state_lonely": "state_na4",
            },
            inplace=True,
        )

        # Create new FLAG column that combines W1 and W2 for trait and state data
        df_emotions_traits["flag_susp_traits"] = df_emotions_traits[
            "outlier_participant_s2w1_t1"
        ].fillna(df_emotions_traits["outlier_participant_s2w2_t1"])
        df_emotions_states["flag_susp_states"] = df_emotions_states[
            "outlier_participant_s2w1"
        ].fillna(df_emotions_states["outlier_participant_s2w2"])

        # add variable to separate ftf and cmc
        df_emotions_states["interaction_medium_binary"] = df_emotions_states[
            "communication"
        ].apply(lambda x: 1 if isinstance(x, (int, float)) and x == 1 else 0)
        tmp_dct_states["emotions_state_lst"][0].append("interaction_medium_binary")

        result_dct["emotions"] = [
            df_emotions_traits[tmp_dct_traits["emotions_trait_lst"]],
            df_emotions_states[tmp_dct_states["emotions_state_lst"][0]],
        ]

    ### Specific processing for Coco UT
    # Default values
    df_coco_ut_traits = None
    df_coco_ut_states = None
    if "coco_ut" in config["samples_for_analysis"]:
        # Coco UT1
        coco_ut1_path = os.path.join(raw_data_path, 'coco_ut1')
        df_coco_ut1_demographics = pd.read_csv(
            os.path.join(coco_ut1_path, "demographics_fall_anonymized.csv"),
            delimiter=",",
            nrows=nrows,
        )  # TODO: Voting state?
        df_coco_ut1_demographics = df_coco_ut1_demographics.rename(
            columns={"pID": "id"}
        )
        df_coco_ut1_personality = pd.read_csv(
            os.path.join(coco_ut1_path, "personality_fall_anonymized.csv"),
            delimiter=",",
            nrows=nrows,
        )
        df_coco_ut1_personality = df_coco_ut1_personality.rename(columns={"pID": "id"})
        df_coco_ut1_presurvey = pd.read_csv(
            os.path.join(coco_ut1_path, "01_presurvey_coco_ut1.csv"),
            delimiter=",", nrows=nrows
        )
        df_coco_ut1_postsurvey = pd.read_csv(
            os.path.join(coco_ut1_path, "06_postsurvey_coco_ut1.csv"),
            delimiter=",",
            nrows=nrows,
        )

        # Add further questionnaire data from other surveys not included in the main data source
        df_coco_ut1_anxiety = pd.read_csv(
            os.path.join(coco_ut1_path, "anxiety_fall_anonymized.csv"),
            nrows=nrows
        )
        df_coco_ut1_anxiety = df_coco_ut1_anxiety.rename(columns={"pID": "id"})
        df_coco_ut1_depression = pd.read_csv(
            os.path.join(coco_ut1_path, "depression_fall_anonymized.csv"),
            nrows=nrows
        )
        df_coco_ut1_depression = df_coco_ut1_depression.rename(columns={"pID": "id"})
        df_coco_ut1_optimism = pd.read_csv(
            os.path.join(coco_ut1_path, "optimism_fall_anonymized.csv"), nrows=nrows
        )
        df_coco_ut1_optimism = df_coco_ut1_optimism.rename(columns={"pID": "id"})
        df_coco_ut1_self_esteem = pd.read_csv(
            os.path.join(coco_ut1_path, "self_esteem_fall_anonymized.csv"), nrows=nrows
        )
        df_coco_ut1_self_esteem = df_coco_ut1_self_esteem.rename(columns={"pID": "id"})

        # Choose different suffixes for possible duplicate columns to prevent merge problems
        df_coco_ut1_traits = (
            df_coco_ut1_demographics.merge(
                df_coco_ut1_personality, on="id", suffixes=(None, "_pers")
            )
            .merge(df_coco_ut1_presurvey, on="id", suffixes=(None, "_pre"))
            .merge(
                df_coco_ut1_postsurvey, on="id", how="left", suffixes=(None, "_post")
            )
            .merge(df_coco_ut1_anxiety, on="id", how="left", suffixes=(None, "_anx"))
            .merge(df_coco_ut1_depression, on="id", how="left", suffixes=(None, "_dep"))
            .merge(df_coco_ut1_optimism, on="id", how="left", suffixes=(None, "_opt"))
            .merge(
                df_coco_ut1_self_esteem, on="id", how="left", suffixes=(None, "_est")
            )
        )

        df_coco_ut1_states = pd.read_csv(
            os.path.join(coco_ut1_path, "04_ema_survey_coco_ut1.csv"), nrows=nrows
        )

        # Coco UT2
        coco_ut2_path = os.path.join(raw_data_path, 'coco_ut2')
        df_coco_ut2_demographics = pd.read_csv(
            os.path.join(coco_ut2_path, "demographics_spring_anonymized.csv"),
            delimiter=",",
            nrows=nrows,
        )
        df_coco_ut2_demographics = df_coco_ut2_demographics.rename(
            columns={"pID": "id"}
        )
        df_coco_ut2_personality = pd.read_csv(
            os.path.join(coco_ut2_path, "personality_spring_anonymized.csv"),
            delimiter=",",
            nrows=nrows,
        )
        df_coco_ut2_personality = df_coco_ut2_personality.rename(columns={"pID": "id"})
        df_coco_ut2_presurvey = pd.read_csv(
            os.path.join(coco_ut2_path, "01_presurvey_coco_ut2.csv"), delimiter=",", nrows=nrows
        )
        df_coco_ut2_postsurvey = pd.read_csv(
            os.path.join(coco_ut2_path, "06_postsurvey_coco_ut2.csv"),
            delimiter=",",
            nrows=nrows,
        )

        # Add further questionnaire data
        df_coco_ut2_anxiety = pd.read_csv(
            os.path.join(coco_ut2_path, "anxiety_spring_anonymized.csv"), nrows=nrows
        )
        df_coco_ut2_anxiety = df_coco_ut2_anxiety.rename(columns={"pID": "id"})
        df_coco_ut2_depression = pd.read_csv(
            os.path.join(coco_ut2_path, "depression_spring_anonymized.csv"), nrows=nrows
        )
        df_coco_ut2_depression = df_coco_ut2_depression.rename(columns={"pID": "id"})
        df_coco_ut2_optimism = pd.read_csv(
            os.path.join(coco_ut2_path, "optimism_spring_anonymized.csv"), nrows=nrows
        )
        df_coco_ut2_optimism = df_coco_ut2_optimism.rename(columns={"pID": "id"})
        df_coco_ut2_self_esteem = pd.read_csv(
            os.path.join(coco_ut2_path, "self_esteem_spring_anonymized.csv"), nrows=nrows
        )
        df_coco_ut2_self_esteem = df_coco_ut2_self_esteem.rename(columns={"pID": "id"})

        # Rename cms to cmq -> consistent with coco_ut1
        pattern = re.compile(r"(cms)(_\w+)")
        column_mappings = {
            col: pattern.sub(r"cmq\2", col)
            for col in df_coco_ut2_presurvey.columns
            if pattern.match(col)
        }
        df_coco_ut2_presurvey.rename(columns=column_mappings, inplace=True)

        # Again, different suffixes
        df_coco_ut2_traits = (
            df_coco_ut2_demographics.merge(
                df_coco_ut2_personality, on="id", suffixes=(None, "_pers")
            )
            .merge(df_coco_ut2_presurvey, on="id", suffixes=(None, "_pre"))
            .merge(
                df_coco_ut2_postsurvey, on="id", how="left", suffixes=(None, "_post")
            )
            .merge(df_coco_ut2_anxiety, on="id", how="left", suffixes=(None, "_anx"))
            .merge(df_coco_ut2_depression, on="id", how="left", suffixes=(None, "_dep"))
            .merge(df_coco_ut2_optimism, on="id", how="left", suffixes=(None, "_opt"))
            .merge(
                df_coco_ut2_self_esteem, on="id", how="left", suffixes=(None, "_est")
            )
        )

        df_coco_ut2_states = pd.read_csv(
            os.path.join(coco_ut2_path, "04_ema_survey_coco_ut2.csv"), nrows=nrows
        )

        df_coco_ut_traits = pd.concat([df_coco_ut1_traits, df_coco_ut2_traits], axis=0)
        df_coco_ut_states = pd.concat([df_coco_ut1_states, df_coco_ut2_states], axis=0)

    # same preprocessing no matter if the data is from CoCo UT1 or CoCo UT2
    if df_coco_ut_traits is not None and df_coco_ut_states is not None:
        # rename bfi
        pattern = re.compile(r"(bfi2)(_\d+)")
        column_mappings = {
            col: pattern.sub(r"\1\2_t1", col)
            for col in df_coco_ut_traits.columns
            if pattern.match(col)
        }
        df_coco_ut_traits.rename(columns=column_mappings, inplace=True)

        # rename ksa
        pattern = re.compile(r"(ksa)(_\d+_\w*)")
        column_mappings = {
            col: pattern.sub(r"ksa3s\2", col)
            for col in df_coco_ut_traits.columns
            if pattern.match(col)
        }
        df_coco_ut_traits.rename(columns=column_mappings, inplace=True)

        # Hotfix, delete columns because of the following substitution
        df_coco_ut_traits.drop(
            ["political_polar_1_t1", "political_polar_2_t1"], axis=1, inplace=True
        )
        # Rename demographics
        df_coco_ut_traits.rename(
            columns={
                "Q24": "age_t1",
                "Q1": "sex_t1",
                "Q33": "area_t1",
                "Q6": "current_living_t1",
                "Q7": "ethnic_group_t1",
                "Q8": "birth_order_t1",
                "Q9": "first_language_t1",
                "Q10": "other_languages_t1",
                # 'Q16': 'classification_t1',
                "Q11": "handedness_t1",
                "Q12": "political_orientation_t1",
                "Q13": "subj_health_rating_t1",
                "Q14": "relationship_status_t1",
                "Q17": "education_mother_t1",
                "Q18": "education_father_t1",
                "Q27": "subj_ses_t1",
                "Q20": "religion_t1",
                "Q21": "religious_affiliation_t1",
                "Q23": "job_t1",
                "demog_ppl_household_t1": "quantity_household_t1",
                "political_polar_7_t1": "political_polar_1_t1",
                "political_polar_8_t1": "political_polar_2_t1",
            },
            inplace=True,
        )

        # Rename bfi (Q1_1 to BFI2_1_t1, etc.)
        for item_nr in range(1, 61):
            pattern = re.compile(rf"^(Q1_)({item_nr})$")
            df_coco_ut_traits.rename(
                columns={
                    col: pattern.sub(f"bfi2_{item_nr}_t1", col)
                    for col in df_coco_ut_traits.columns
                    if pattern.match(col)
                },
                inplace=True,
            )
        # rename psm to wgm
        mapping_policy = {}
        pattern = re.compile(rf"^(psm_)(\w+)(_t1)")
        col_names = [col for col in df_coco_ut_traits.columns if pattern.match(col)]
        for idx, col_name in enumerate(col_names, 1):
            mapping_policy[col_name] = f"wgm_{idx}_t1"
        df_coco_ut_traits.rename(columns=mapping_policy, inplace=True)

        # Rename additional variables from the other sources (anxiety, depression, etc.)
        # anxiety
        for item_nr in range(1, 8):
            df_coco_ut_traits.rename(
                columns={f"Q1_{item_nr}_anx": f"gad7_{item_nr}_t1"}, inplace=True
            )
        # depression
        for item_nr in range(1, 11):
            df_coco_ut_traits.rename(
                columns={f"Q1_{item_nr}_dep": f"cesd_{item_nr}_t1"}, inplace=True
            )
        # optimism
        for item_nr in range(1, 7):
            df_coco_ut_traits.rename(
                columns={f"Q2_{item_nr}": f"lotr_{item_nr}_t1"}, inplace=True
            )
        # self esteem
        for item_nr in range(1, 11):
            df_coco_ut_traits.rename(
                columns={f"Q1_{item_nr}_est": f"self_esteem_{item_nr}_t1"}, inplace=True
            )

        # Optimism (LOT-R) is also assessed in coco_ut2 -> compare and fill missings
        # 10-item version (CoCo UT2) has 4 filler items, therefore the assignment seems strange
        if "coco_ut" in config["samples_for_analysis"]:
            df_coco_ut_traits["lotr_1_t1"] = df_coco_ut_traits["lotr_1_t1"].fillna(
                df_coco_ut_traits["lot_1_t1"]
            )
            df_coco_ut_traits["lotr_2_t1"] = df_coco_ut_traits["lotr_2_t1"].fillna(
                df_coco_ut_traits["lot_3_t1"]
            )
            df_coco_ut_traits["lotr_3_t1"] = df_coco_ut_traits["lotr_3_t1"].fillna(
                df_coco_ut_traits["lot_4_t1"]
            )
            df_coco_ut_traits["lotr_4_t1"] = df_coco_ut_traits["lotr_4_t1"].fillna(
                df_coco_ut_traits["lot_7_t1"]
            )
            df_coco_ut_traits["lotr_5_t1"] = df_coco_ut_traits["lotr_5_t1"].fillna(
                df_coco_ut_traits["lot_9_t1"]
            )
            df_coco_ut_traits["lotr_6_t1"] = df_coco_ut_traits["lotr_6_t1"].fillna(
                df_coco_ut_traits["lot_10_t1"]
            )

        # rename state wb items
        df_coco_ut_states.rename(
            columns={
                "momentary_wellbeing_happy_ema": "state_pa1",
                "momentary_wellbeing_angry_ema": "state_na1",
                "momentary_wellbeing_worried_ema": "state_na2",
                "momentary_wellbeing_sad_ema": "state_na3",
                "momentary_wellbeing_stressed_ema": "state_na4",
                "momentary_wellbeing_lonely_ema": "state_na5",
            },
            inplace=True,
        )

        df_coco_ut_states.rename(
            columns={
                "momentary_context_interaction_ema": "social_interaction",
                "momentary_context_inperson_interact_ema": "interaction_quantity",
            },
            inplace=True,
        )

        # rename column that represents the time point of ESM-survey creation
        df_coco_ut_states.rename(
            columns={"RecordedDateConvert": "created_esm_timepoint"}, inplace=True
        )

        # add variable to separate ftf and cmc
        df_coco_ut_states["interaction_medium_binary"] = df_coco_ut_states[
            "soc_interact_mode_ema"
        ].apply(lambda x: 1 if isinstance(x, str) and "1" in x else 0)
        tmp_dct_states["coco_ut_state_lst"][0].append("interaction_medium_binary")

        # Add interaction_quantity to coco_ut, because it is added when suppl analysis is ftf
        tmp_dct_states["coco_ut_state_lst"][0].append("interaction_quantity")

        # interaction quantity was an open question, currently got string values -> further process this
        df_coco_ut_states["interaction_quantity"] = (
            df_coco_ut_states["interaction_quantity"].str.lower().str.strip()
        )
        df_coco_ut_states["interaction_quantity"] = df_coco_ut_states[
            "interaction_quantity"
        ].str.replace("+", "", regex=False)
        # Dictionary for string to number mapping, using 20 for "a lot" is a bit arbitrary, but ok
        number_map = {
            "none": 0,
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "a lot": 20,
            "lots": 20,
            "too many": 20,
            "5-10": 7,
            "many": 20,
            "00": 0,
            "-": 0,
            '"': 0,
        }
        # neglecting unclear cases or strings that were just answered once
        df_coco_ut_states["interaction_quantity"] = df_coco_ut_states[
            "interaction_quantity"
        ].apply(lambda x: convert_to_numeric(x, number_map))
        df_coco_ut_traits = df_coco_ut_traits.reset_index(drop=True)
        df_coco_ut_states = df_coco_ut_states.reset_index(drop=True)
        result_dct["coco_ut"] = [
            df_coco_ut_traits[tmp_dct_traits["coco_ut_trait_lst"]],
            df_coco_ut_states[tmp_dct_states["coco_ut_state_lst"][0]],
        ]
    return result_dct


def preprocess_country_data(config_path):
    """
    This function preprocesses the raw country data from Hofstede and the UN dataset. It stores the data for
    further use in preprocessor.

    Args:
        config_path: str, relative path to the YAML config
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    country_var_cfg = config["country_variables"]
    country_var_raw_data_path = country_var_cfg["general_raw_data_path"]

    if "hofstede" in country_var_cfg["samples"]:
        df_hofstede = pd.read_csv(
            os.path.join(country_var_raw_data_path, "hofstede/hofstede_country_variables.csv"),
            delimiter=";",
        )

        # Preprocessing / Format
        df_hofstede = df_hofstede.applymap(lambda x: np.nan if x == "#NULL!" else x)
        df_hofstede["country"] = df_hofstede.country.str.lower()
        df_hofstede = df_hofstede.set_index("country", drop=True)
        df_hofstede.drop("ctr", axis=1, inplace=True)

        # Rename columns and create new columns for countries that are aggregated under a region in the Hofstede dataset
        # Hofstedes data covers almost all countries that remain in the coco int dataset after filtering
        df_hofstede.rename(
            index={
                "czech rep": "czech_republic",
                "u.s.a.": "usa",
                "great britain": "united_kingdom",
            },
            inplace=True,
        )
        df_hofstede.loc["south africa"] = df_hofstede.loc["south africa"].fillna(
            df_hofstede.loc["south africa white"]
        )
        df_hofstede.loc["south_africa"] = df_hofstede.loc["south africa"]
        df_hofstede.loc["angola"] = df_hofstede.loc["south africa"]
        df_hofstede.loc["namibia"] = df_hofstede.loc["south africa"]

        df_hofstede.loc["kuwait"] = df_hofstede.loc["arab countries"]
        df_hofstede.loc["united_arab_emirates"] = df_hofstede.loc["arab countries"]

        # Drop all rows with missing values on at least one of the 6 dimensions
        df_hofstede.dropna(axis=0, inplace=True)

        # Store data
        if country_var_cfg["store_data"]:
            file_path = country_var_cfg["store_path"]
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            with open(os.path.join(file_path, "hofstede_preprocessed.pkl"), "wb") as f:
                pickle.dump(df_hofstede, f)

    if "un" in country_var_cfg["samples"]:
        df_lst = list()
        for subsample in country_var_cfg["un"]["un_subsamples"]:
            df = pd.read_csv(
                os.path.join(country_var_raw_data_path, f"un/{subsample['name']}.csv"),
                delimiter=",",
                skiprows=1,
            )
            pivot_column_name_mapping = subsample["pivot_column_name_mapping"]
            value_column_name_mapping = subsample["value_column_name_mapping"]
            pivot_columns = subsample["pivot_columns"]

            # Preprocess all un subsamples
            df.columns = [str(col_name).lower().strip() for col_name in df.columns]
            df.rename(columns=pivot_column_name_mapping, inplace=True)

            # correct GDP for COVID-19 effects, use 2019 instead of more recent data
            if subsample["name"] == "un_gdp":
                df = df[df["year"] == 2019]
            else:
                max_years_per_var = df.groupby(subsample["pivot_columns"])[
                    "year"
                ].transform("max")
                year_mask = df["year"] == max_years_per_var
                df = df[year_mask]
            df_pivot = df.pivot(index="country", columns=pivot_columns, values="value")
            df_pivot.rename(columns=value_column_name_mapping, inplace=True)
            df_pivot.index = df_pivot.index.map(str.lower)
            df_pivot = df_pivot[
                [
                    col
                    for col in df_pivot.columns
                    if col in value_column_name_mapping.values()
                ]
            ]

            # adjust country names
            df_pivot.index = [string.replace(" ", "_") for string in df_pivot.index]
            df_pivot.rename(
                index={
                    "iran_(islamic_republic_of)": "iran",
                    "united_states_of_america": "usa",
                    "viet_nam": "vietnam",
                    "türkiye": "turkey",
                    "czechia": "czech_republic",
                    "republic_of_korea": "korea_south",
                    "united_rep._of_tanzania": "tanzania",
                },
                inplace=True,
            )
            df_lst.append(df_pivot)
        df_un = pd.concat(df_lst, axis=1)

        # Preprocess joined df
        for col in df_un.columns:
            df_un[col] = df_un[col].astype(str).str.replace(",", "").astype(float)
        df_un.dropna(axis=0, inplace=True)

        # Store data
        if country_var_cfg["store_data"]:
            file_path = country_var_cfg["store_path"]
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            with open(os.path.join(file_path, "un_preprocessed.pkl"), "wb") as f:
                pickle.dump(df_un, f)

    if "coco_mixed" in country_var_cfg["samples"]:
        df_coco_mixed = pd.read_csv(
            os.path.join(country_var_raw_data_path, "coco_mixed/coco_country_mixed.txt"),
            delim_whitespace=True,
        )
        df_coco_mixed.set_index("country", inplace=True)
        # convert column where decimal signs are commas instead of points -> causes problems otherwise
        cols_to_convert = ["religion_idx", "psm_eco_idx", "psm_pol_idx"]
        df_coco_mixed = df_coco_mixed.drop("kuwait", axis=0)
        df_coco_mixed[cols_to_convert] = df_coco_mixed[cols_to_convert].applymap(
            lambda x: float(x.replace(",", "."))
        )

        # Store data
        if country_var_cfg["store_data"]:
            file_path = country_var_cfg["store_path"]
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            with open(
                os.path.join(file_path, "coco_mixed_preprocessed.pkl"), "wb"
            ) as f:
                pickle.dump(df_coco_mixed, f)
    print()


def convert_to_numeric(val, number_map):
    """
    This function converts strings representing certain numbers to numeric values. This is used to convert
    the strings representing the number of interaction partners in CoCo UT to numeric values.

    Args:
        val: str, value to be converted
        number_map: Dict, defining which strings are converted to which numbers

    Returns:
        int(val): If val ist just a numeric string (e.g., "6"), return the number
        number_map(val): If val ist not just a numeric string (e.g., "6"), return the value corresponding to the key
            "val" in "number_map"
        None: Return None if val is not in number_map
    """
    try:
        return int(val)
    except:
        if val in number_map:
            return number_map[val]
        return None
