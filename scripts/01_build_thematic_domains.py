#!/usr/bin/env python3
"""
Build thematic_domains.json from CES 2021 Stata variable labels.

Reads variable labels directly from the .dta file (no hardcoding),
assigns each kept cps21_ variable to a thematic domain, and writes:
    src/article_silicon_sampling_quebec/thematic_domains.json

Schema:
    {
        "variable_name": {
            "label": "<Stata variable label>",
            "domain": "<thematic domain>"
        },
        ...
    }

Thematic domains:
    - vote_behaviour       : vote choice, party lean, past vote, likelihood
    - party_leader_eval    : party/leader/candidate ratings, leader traits
    - issue_ownership      : which party best handles which issue, campaign issue
    - political_attitudes  : dem satisfaction, govt trust, party ID, left-right scale
    - economic_attitudes   : economic retrospection, spending preferences, env-economy tradeoffs
    - social_attitudes     : immigration, discrimination, end-of-life, cannabis, reconciliation
    - national_identity    : Quebec sovereignty, federal/provincial identity strength
    - covid                : COVID relief, vaccine, health measure satisfaction

SES variables (yob, province, education, gender, language) are excluded —
they are not attitude questions and do not need a thematic domain.
"""

import json
import logging
from pathlib import Path

from pandas.io.stata import StataReader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping: variable_name -> domain
# Built from inspection of Stata variable labels (see 01_build_thematic_domains.py
# header and data/raw/ces_2021/ces_2021_codebook_questions.md).
# Only variables actually kept in the cleaned dataset are included
# (i.e. cps21_ attitude questions, no _DO_/_t/captcha/TEXT/admin).
# ---------------------------------------------------------------------------
DOMAIN_MAP: dict[str, str] = {
    # --- vote_behaviour ---
    "cps21_v_likely":           "vote_behaviour",
    "cps21_howvote2":           "vote_behaviour",
    "cps21_votechoice":         "vote_behaviour",
    "cps21_vote_unlikely":      "vote_behaviour",
    "cps21_v_advance":          "vote_behaviour",
    "cps21_vote_lean":          "vote_behaviour",
    "cps21_2nd_choice":         "vote_behaviour",
    "cps21_not_vote_for_1":     "vote_behaviour",
    "cps21_not_vote_for_2":     "vote_behaviour",
    "cps21_not_vote_for_3":     "vote_behaviour",
    "cps21_not_vote_for_4":     "vote_behaviour",
    "cps21_not_vote_for_5":     "vote_behaviour",
    "cps21_not_vote_for_6":     "vote_behaviour",
    "cps21_not_vote_for_7":     "vote_behaviour",
    "cps21_not_vote_for_8":     "vote_behaviour",
    "cps21_not_vote_for_w_1":   "vote_behaviour",
    "cps21_not_vote_for_w_2":   "vote_behaviour",
    "cps21_not_vote_for_w_3":   "vote_behaviour",
    "cps21_not_vote_for_w_4":   "vote_behaviour",
    "cps21_not_vote_for_w_5":   "vote_behaviour",
    "cps21_vote_2019":          "vote_behaviour",
    "cps21_spoil":              "vote_behaviour",
    "cps21_duty_choice":        "vote_behaviour",
    "cps21_comfort1":           "vote_behaviour",
    "cps21_comfort2":           "vote_behaviour",
    "cps21_comfort3":           "vote_behaviour",
    "cps21_debate_fr":          "vote_behaviour",
    "cps21_debate_fr2":         "vote_behaviour",
    "cps21_debate_en":          "vote_behaviour",

    # --- party_leader_eval ---
    "cps21_party_rating_23":    "party_leader_eval",
    "cps21_party_rating_24":    "party_leader_eval",
    "cps21_party_rating_25":    "party_leader_eval",
    "cps21_party_rating_26":    "party_leader_eval",
    "cps21_party_rating_27":    "party_leader_eval",
    "cps21_party_rating_29":    "party_leader_eval",
    "cps21_lead_rating_23":     "party_leader_eval",
    "cps21_lead_rating_24":     "party_leader_eval",
    "cps21_lead_rating_25":     "party_leader_eval",
    "cps21_lead_rating_26":     "party_leader_eval",
    "cps21_lead_rating_27":     "party_leader_eval",
    "cps21_lead_rating_29":     "party_leader_eval",
    "cps21_cand_rating_23":     "party_leader_eval",
    "cps21_cand_rating_24":     "party_leader_eval",
    "cps21_cand_rating_25":     "party_leader_eval",
    "cps21_cand_rating_26":     "party_leader_eval",
    "cps21_cand_rating_27":     "party_leader_eval",
    "cps21_lead_int_1":         "party_leader_eval",
    "cps21_lead_int_2":         "party_leader_eval",
    "cps21_lead_int_3":         "party_leader_eval",
    "cps21_lead_int_4":         "party_leader_eval",
    "cps21_lead_int_5":         "party_leader_eval",
    "cps21_lead_int_6":         "party_leader_eval",
    "cps21_lead_int_7":         "party_leader_eval",
    "cps21_lead_strong_1":      "party_leader_eval",
    "cps21_lead_strong_2":      "party_leader_eval",
    "cps21_lead_strong_3":      "party_leader_eval",
    "cps21_lead_strong_4":      "party_leader_eval",
    "cps21_lead_strong_5":      "party_leader_eval",
    "cps21_lead_strong_6":      "party_leader_eval",
    "cps21_lead_strong_7":      "party_leader_eval",
    "cps21_lead_trust_1":       "party_leader_eval",
    "cps21_lead_trust_2":       "party_leader_eval",
    "cps21_lead_trust_3":       "party_leader_eval",
    "cps21_lead_trust_4":       "party_leader_eval",
    "cps21_lead_trust_5":       "party_leader_eval",
    "cps21_lead_trust_6":       "party_leader_eval",
    "cps21_lead_trust_7":       "party_leader_eval",
    "cps21_lead_cares_1":       "party_leader_eval",
    "cps21_lead_cares_2":       "party_leader_eval",
    "cps21_lead_cares_3":       "party_leader_eval",
    "cps21_lead_cares_4":       "party_leader_eval",
    "cps21_lead_cares_5":       "party_leader_eval",
    "cps21_lead_cares_6":       "party_leader_eval",
    "cps21_lead_cares_7":       "party_leader_eval",
    "cps21_lib_promises":       "party_leader_eval",

    # --- issue_ownership ---
    "cps21_imp_iss":            "issue_ownership",
    "cps21_imp_iss_party":      "issue_ownership",
    "cps21_imp_loc_iss":        "issue_ownership",
    "cps21_imp_loc_iss_p":      "issue_ownership",
    "cps21_camp_issue":         "issue_ownership",
    "cps21_issue_handle_1":     "issue_ownership",
    "cps21_issue_handle_2":     "issue_ownership",
    "cps21_issue_handle_3":     "issue_ownership",
    "cps21_issue_handle_4":     "issue_ownership",
    "cps21_issue_handle_5":     "issue_ownership",
    "cps21_issue_handle_6":     "issue_ownership",
    "cps21_issue_handle_7":     "issue_ownership",
    "cps21_issue_handle_8":     "issue_ownership",
    "cps21_issue_handle_9":     "issue_ownership",
    "cps21_outcome_most":       "issue_ownership",
    "cps21_outcome_least":      "issue_ownership",

    # --- political_attitudes ---
    "cps21_demsat":             "political_attitudes",
    "cps21_fed_gov_sat":        "political_attitudes",
    "cps21_prov_gov_sat":       "political_attitudes",
    "cps21_govt_confusing":     "political_attitudes",
    "cps21_govt_say":           "political_attitudes",
    "cps21_pol_eth":            "political_attitudes",
    "cps21_interest_gen_1":     "political_attitudes",
    "cps21_interest_elxn_1":    "political_attitudes",
    "cps21_lr_scale_bef_1":     "political_attitudes",
    "cps21_lr_parties_1":       "political_attitudes",
    "cps21_lr_parties_2":       "political_attitudes",
    "cps21_lr_parties_3":       "political_attitudes",
    "cps21_lr_parties_4":       "political_attitudes",
    "cps21_lr_parties_5":       "political_attitudes",
    "cps21_lr_parties_7":       "political_attitudes",
    "cps21_fed_id":             "political_attitudes",
    "cps21_fed_id_str":         "political_attitudes",
    "cps21_prov_id":            "political_attitudes",
    "cps21_prov_id_str":        "political_attitudes",
    "cps21_minority_gov":       "political_attitudes",
    "cps21_pos_mailtrust":      "political_attitudes",
    "cps21_pos_fptp":           "political_attitudes",
    "cps21_most_seats_1":       "political_attitudes",
    "cps21_most_seats_2":       "political_attitudes",
    "cps21_most_seats_3":       "political_attitudes",
    "cps21_most_seats_4":       "political_attitudes",
    "cps21_most_seats_5":       "political_attitudes",
    "cps21_win_local_1":        "political_attitudes",
    "cps21_win_local_2":        "political_attitudes",
    "cps21_win_local_3":        "political_attitudes",
    "cps21_win_local_4":        "political_attitudes",
    "cps21_win_local_5":        "political_attitudes",
    "cps21__candidateref":      "political_attitudes",
    "cps21_candidate_imag":     "political_attitudes",
    "cps21_news_cons":          "political_attitudes",
    "cps21_premier_name":       "political_attitudes",
    "cps21_finmin_name":        "political_attitudes",
    "cps21_govgen_name":        "political_attitudes",
    "cps21_volunteer":          "political_attitudes",

    # --- economic_attitudes ---
    "cps21_spend_educ":         "economic_attitudes",
    "cps21_spend_env":          "economic_attitudes",
    "cps21_spend_just_law":     "economic_attitudes",
    "cps21_spend_defence":      "economic_attitudes",
    "cps21_spend_imm_min":      "economic_attitudes",
    "cps21_spend_rec_indi":     "economic_attitudes",
    "cps21_spend_afford_h":     "economic_attitudes",
    "cps21_spend_nation_c":     "economic_attitudes",
    "cps21_pos_carbon":         "economic_attitudes",
    "cps21_pos_energy":         "economic_attitudes",
    "cps21_pos_envreg":         "economic_attitudes",
    "cps21_pos_jobs":           "economic_attitudes",
    "cps21_pos_subsid":         "economic_attitudes",
    "cps21_pos_trade":          "economic_attitudes",
    "cps21_econ_retro":         "economic_attitudes",
    "cps21_econ_fed_bette":     "economic_attitudes",
    "cps21_own_fin_retro":      "economic_attitudes",
    "cps21_ownfinanc_fed":      "economic_attitudes",
    "cps21_own_fin_future":     "economic_attitudes",

    # --- social_attitudes ---
    "cps21_imm":                "social_attitudes",
    "cps21_refugees":           "social_attitudes",
    "cps21_groupdiscrim_1":     "social_attitudes",
    "cps21_groupdiscrim_2":     "social_attitudes",
    "cps21_groupdiscrim_3":     "social_attitudes",
    "cps21_groupdiscrim_4":     "social_attitudes",
    "cps21_groupdiscrim_5":     "social_attitudes",
    "cps21_groupdiscrim_6":     "social_attitudes",
    "cps21_groupdiscrim_7":     "social_attitudes",
    "cps21_groupdiscrim_8":     "social_attitudes",
    "cps21_pos_life":           "social_attitudes",
    "cps21_pos_cannabis":       "social_attitudes",
    "cps21_residential_2a":     "social_attitudes",
    "cps21_residential_2b":     "social_attitudes",
    "cps21_residential_2c":     "social_attitudes",
    "cps21_residential_2d":     "social_attitudes",
    "cps21_groups_therm_1":     "social_attitudes",
    "cps21_groups_therm_2":     "social_attitudes",
    "cps21_groups_therm_3":     "social_attitudes",
    "cps21_groups_therm_4":     "social_attitudes",
    "cps21_groups_therm_6":     "social_attitudes",

    # --- national_identity ---
    "cps21_quebec_sov":         "national_identity",

    # --- covid ---
    "cps21_covid_liberty":      "covid",
    "cps21_covidrelief__1":     "covid",
    "cps21_covidrelief__2":     "covid",
    "cps21_covidrelief__3":     "covid",
    "cps21_covidrelief__4":     "covid",
    "cps21_covidrelief__5":     "covid",
    "cps21_covidrelief__6":     "covid",
    "cps21_covidrelief__7":     "covid",
    "cps21_covidrelief__8":     "covid",
    "cps21_covidrelief__9":     "covid",
    "cps21_covid_sat_1":        "covid",
    "cps21_covid_sat_2":        "covid",
    "cps21_covid_sat_3":        "covid",
    "cps21_vaccine_mandat_1":   "covid",
    "cps21_vaccine_mandat_2":   "covid",
    "cps21_vaccine_mandat_3":   "covid",
    "cps21_vaccine1":           "covid",
    "cps21_vaccine2":           "covid",
    "cps21_vaccine3":           "covid",
}


def build_thematic_domains(stata_path: str, output_path: str) -> None:
    """Read Stata labels and write thematic_domains.json."""
    logger.info(f"Reading variable labels from {stata_path}...")
    reader = StataReader(stata_path, convert_categoricals=False)
    var_labels = reader.variable_labels()

    output: dict[str, dict] = {}
    unmatched: list[str] = []

    for var_name, domain in DOMAIN_MAP.items():
        label = var_labels.get(var_name, "")
        if not label:
            unmatched.append(var_name)
        output[var_name] = {
            "label": label,
            "domain": domain,
        }

    if unmatched:
        logger.warning(
            f"{len(unmatched)} variables in DOMAIN_MAP not found in Stata file: "
            f"{unmatched}"
        )

    # Summary
    from collections import Counter
    counts = Counter(v["domain"] for v in output.values())
    logger.info("Domain summary:")
    for domain, n in sorted(counts.items()):
        logger.info(f"  {domain:25s} {n:3d} variables")
    logger.info(f"  {'TOTAL':25s} {sum(counts.values()):3d} variables")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

    logger.info(f"Written: {output_path}")


def main() -> None:
    stata_path = "data/raw/ces_2021/ces_2021.dta"
    output_path = "src/article_silicon_sampling_quebec/thematic_domains.json"
    build_thematic_domains(stata_path, output_path)


if __name__ == "__main__":
    main()
