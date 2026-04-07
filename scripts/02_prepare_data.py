#!/usr/bin/env python3
"""
Prepare CES 2021 data for silicon sampling pipeline.

Run AFTER: 01_build_thematic_domains.py

Processing steps:
1. Filter Quebec respondents only (cps21_province == 11)
2. Select relevant variables (SES + attitude questions)
3. Transform numeric values to text labels via Stata value labels
4. Add thematic_domain column to questions metadata

Outputs:
    data/processed/questions.parquet   — one row per question (variable metadata)
    data/processed/respondents.parquet — respondents × variables matrix (text labels)
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from pandas.io.stata import StataReader

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

LANGUAGE_DUMMY_COLS = [f"cps21_language_{i}" for i in range(1, 19)]

MANUAL_PARENT_LABELS: dict[str, dict[str, str]] = {
    "cps21_groupdiscrim": {
        "en": "How much discrimination is there in Canada against the following groups?",
        "fr": "À quel point les groupes suivants font-ils face à de la discrimination au Canada?",
    },
    "cps21_covid_sat": {
        "en": "How satisfied are you with how each of the following have handled the coronavirus outbreak?",
        "fr": "À quel point êtes-vous satisfait(e) de la façon dont les gouvernements suivants ont géré la pandémie de coronavirus?",
    },
    "cps21_vaccine_mandat": {
        "en": "Should vaccination be required to:",
        "fr": "La vaccination devrait être requise pour:",
    },
    "cps21_groups_therm": {
        "en": "How do you feel about the following groups?",
        "fr": "Que pensez-vous des différents groupes ci-dessous?",
    },
    "cps21_lr_parties": {
        "en": "Where would you place the federal political parties on a scale from left (0) to right (10)?",
        "fr": "Où placeriez-vous les partis politiques fédéraux sur une échelle de 0 (gauche) à 10 (droite)?",
    },
    "cps21_most_seats": {
        "en": "How likely is each party below to win the most House of Commons seats?",
        "fr": "Quel parti a le plus de chances de gagner le plus grand nombre de sièges à la Chambre des communes?",
    },
    "cps21_win_local": {
        "en": "How likely is each party to win the seat in your own local riding?",
        "fr": "Quel parti a le plus de chances de gagner le siège de votre circonscription?",
    },
    "cps21_covidrelief": {
        "en": "Applied for any of the following COVID relief programs? (Select all that apply)",
        "fr": "Avez-vous demandé l'une de ces prestations d'urgence en lien avec la COVID-19? (Sélectionnez toutes celles qui s'appliquent)",
    },
}

MANUAL_CHILD_LABELS: dict[str, dict[str, str]] = {
    "cps21_groupdiscrim_1": {"en": "Indigenous Peoples", "fr": "Peuples autochtones"},
    "cps21_groupdiscrim_2": {"en": "Black people or people of colour", "fr": "Personnes noires ou racisées"},
    "cps21_groupdiscrim_3": {"en": "Immigrants", "fr": "Immigrants"},
    "cps21_groupdiscrim_4": {"en": "Women", "fr": "Femmes"},
    "cps21_groupdiscrim_5": {"en": "Men", "fr": "Hommes"},
    "cps21_groupdiscrim_6": {"en": "Gays and lesbians", "fr": "Personnes gaies et lesbiennes"},
    "cps21_groupdiscrim_7": {"en": "Transgender people", "fr": "Personnes transgenres"},
    "cps21_groupdiscrim_8": {"en": "White people", "fr": "Personnes blanches"},
    "cps21_covid_sat_1": {"en": "Federal government", "fr": "Gouvernement fédéral"},
    "cps21_covid_sat_2": {"en": "Provincial government", "fr": "Gouvernement provincial"},
    "cps21_covid_sat_3": {"en": "Local public health authorities", "fr": "Autorités locales de santé publique"},
    "cps21_vaccine_mandat_1": {"en": "Travel by air or rail in Canada", "fr": "Voyager en avion ou en train au Canada"},
    "cps21_vaccine_mandat_2": {"en": "Go to a bar or restaurant", "fr": "Aller dans un bar ou un restaurant"},
    "cps21_vaccine_mandat_3": {"en": "Work in a hospital", "fr": "Travailler dans un hôpital"},
    "cps21_groups_therm_1": {"en": "Racial minorities", "fr": "Minorités racisées"},
    "cps21_groups_therm_2": {"en": "Immigrants", "fr": "Immigrants"},
    "cps21_groups_therm_3": {"en": "Francophones", "fr": "Francophones"},
    "cps21_groups_therm_4": {"en": "Indigenous Peoples", "fr": "Peuples autochtones"},
    "cps21_groups_therm_6": {"en": "Feminists", "fr": "Féministes"},
    "cps21_lr_parties_1": {"en": "Liberal Party", "fr": "Parti libéral"},
    "cps21_lr_parties_2": {"en": "Conservative Party", "fr": "Parti conservateur"},
    "cps21_lr_parties_3": {"en": "NDP", "fr": "NPD"},
    "cps21_lr_parties_4": {"en": "Bloc Québécois", "fr": "Bloc québécois"},
    "cps21_lr_parties_5": {"en": "Green Party", "fr": "Parti vert"},
    "cps21_lr_parties_7": {"en": "People's Party", "fr": "Parti populaire"},
    "cps21_most_seats_1": {"en": "Liberal Party", "fr": "Parti libéral"},
    "cps21_most_seats_2": {"en": "Conservative Party", "fr": "Parti conservateur"},
    "cps21_most_seats_3": {"en": "NDP", "fr": "NPD"},
    "cps21_most_seats_4": {"en": "Bloc Québécois", "fr": "Bloc québécois"},
    "cps21_most_seats_5": {"en": "Green Party", "fr": "Parti vert"},
    "cps21_win_local_1": {"en": "Liberal Party", "fr": "Parti libéral"},
    "cps21_win_local_2": {"en": "Conservative Party", "fr": "Parti conservateur"},
    "cps21_win_local_3": {"en": "NDP", "fr": "NPD"},
    "cps21_win_local_4": {"en": "Bloc Québécois", "fr": "Bloc québécois"},
    "cps21_win_local_5": {"en": "Green Party", "fr": "Parti vert"},
    "cps21_covidrelief__1": {"en": "Canada Emergency Response Benefit (CERB)", "fr": "Prestation canadienne d'urgence (PCU)"},
    "cps21_covidrelief__2": {"en": "Canada Emergency Student Benefit (CESB)", "fr": "Prestation canadienne d'urgence pour les étudiants (PCUE)"},
    "cps21_covidrelief__3": {"en": "Canada Recovery Benefit (CRB)", "fr": "Prestation canadienne de la relance économique (PCRE)"},
    "cps21_covidrelief__4": {"en": "Canada Recovery Sickness Benefit (CRSB)", "fr": "Prestation canadienne de maladie pour la relance économique (PCMRE)"},
    "cps21_covidrelief__5": {"en": "Canada Recovery Caregiving Benefit (CRCB)", "fr": "Prestation canadienne de la relance économique pour proches aidants (PCREPA)"},
    "cps21_covidrelief__6": {"en": "Canada Emergency Wage Subsidy (CEWS)", "fr": "Subvention salariale d'urgence du Canada (SSUC)"},
    "cps21_covidrelief__7": {"en": "Canada Emergency Rent Subsidy (CERS)", "fr": "Subvention d'urgence du Canada pour le loyer (SUCL)"},
    "cps21_covidrelief__8": {"en": "None", "fr": "Aucune"},
    "cps21_covidrelief__9": {"en": "Don't know/ Prefer not to answer", "fr": "Je ne sais pas/Préfère ne pas répondre"},
}

# Manual FR option lists for variables whose codebook FR entry is missing or
# garbled (e.g. cps21_groupdiscrim_* where options_fr is null in the JSONL).
# Keys are parent variable names; the list must mirror the EN options order/codes.
MANUAL_OPTIONS_FR: dict[str, list[str]] = {
    "cps21_groupdiscrim": [
        "1: Énormément",
        "2: Beaucoup",
        "3: Modérément",
        "4: Un peu",
        "5: Pas du tout",
        "6: Je ne sais pas/Préfère ne pas répondre",
    ],
}

# For some variables the DTA stores value labels in a different case than the
# codebook options used to build the en_to_fr localization map in downstream
# scripts.  Map the exact DTA strings to the canonical codebook strings.
VALUE_LABEL_NORMALIZATIONS: dict[str, dict[str, str]] = {
    "cps21_econ_retro": {
        "got better": "Got better",
        "stayed about the same": "Stayed about the same",
        "got worse": "Got worse",
        "don\u2019t know": "Don't know",
        "dont know": "Don't know",
        "prefer not to answer": "Prefer not to answer",
    },
}

MANUAL_LABEL_OVERRIDES: dict[str, dict[str, str]] = {
    "cps21_fed_id_str": {
        "en": "How strongly do you feel attached to your federal party?",
        "fr": "À quel point vous sentez-vous proche de votre parti fédéral?",
    },
    "cps21_prov_id_str": {
        "en": "How strongly do you feel attached to your provincial party?",
        "fr": "À quel point vous sentez-vous proche de votre parti provincial?",
    },
    "cps21_spend_just_law": {
        "en": "How much should the federal government spend on justice and law enforcement?",
        "fr": "Combien le gouvernement fédéral devrait-il dépenser pour la justice et l'application de la loi?",
    },
    "cps21_vaccine3": {
        "en": "If not vaccinated, are you:",
        "fr": "Si vous n'êtes pas vacciné(e), êtes-vous:",
    },
    "cps21_fed_gov_sat": {
        "fr": "Quel est votre niveau de satisfaction quant à la performance du gouvernement fédéral sous Justin Trudeau?",
    },
    "cps21_prov_gov_sat": {
        "fr": "Quel est votre niveau de satisfaction quant à la performance du gouvernement provincial?",
    },
    "cps21_spend_educ": {
        "fr": "Combien le gouvernement fédéral devrait-il dépenser en éducation?",
    },
    "cps21_spend_env": {
        "fr": "Combien le gouvernement fédéral devrait-il dépenser en environnement?",
    },
    "cps21_spend_defence": {
        "fr": "Combien le gouvernement fédéral devrait-il dépenser en défense?",
    },
    "cps21_spend_imm_min": {
        "fr": "Combien le gouvernement fédéral devrait-il dépenser pour les immigrants et les minorités?",
    },
    "cps21_spend_rec_indi": {
        "fr": "Combien le gouvernement fédéral devrait-il dépenser pour la réconciliation avec les peuples autochtones?",
    },
    "cps21_spend_afford_h": {
        "fr": "Combien le gouvernement fédéral devrait-il dépenser pour le logement abordable?",
    },
    "cps21_spend_nation_c": {
        "fr": "Combien le gouvernement fédéral devrait-il dépenser pour un système national de garderies?",
    },
    "cps21_fed_id": {
        "en": "In federal politics, do you usually think of yourself as:",
        "fr": "En politique fédérale, vous considérez-vous habituellement comme:",
    },
    "cps21_prov_id": {
        "en": "In provincial politics, do you usually think of yourself as:",
        "fr": "En politique provinciale, vous considérez-vous habituellement comme:",
    },
    "cps21_minority_gov": {
        "en": "Do you think minority governments are a good thing or a bad thing?",
        "fr": "Pensez-vous que les gouvernements minoritaires sont une bonne ou une mauvaise chose?",
    },
    "cps21_v_likely": {
        "fr": "Le jour de l'élection, quelle est la probabilité que vous votiez?",
    },
    "cps21_econ_retro": {
        "en": "Over the past year, has Canada's economy gotten better, stayed about the same, or gotten worse?",
        "fr": "Depuis un an, l'économie canadienne s'est-elle améliorée, est-elle restée à peu près la même, ou s'est-elle détériorée?",
    },
    "cps21_econ_fed_bette": {
        "en": "Have the policies of the federal government made Canada's economy better, worse, or not made much difference?",
        "fr": "Les politiques du gouvernement fédéral ont-elles amélioré l'économie canadienne, l'ont-elles détériorée, ou n'ont-elles pas changé grand-chose?",
    },
    "cps21_own_fin_retro": {
        "en": "Over the past year, has your financial situation gotten better, stayed about the same, or gotten worse?",
        "fr": "Pendant la dernière année, votre situation financière s'est-elle améliorée, est-elle restée à peu près la même, ou s'est-elle détériorée?",
    },
    "cps21_ownfinanc_fed": {
        "en": "Have the policies of the federal government made your financial situation better, worse, or not made much difference?",
        "fr": "Les politiques du gouvernement fédéral ont-elles amélioré votre situation financière, l'ont-elles détériorée, ou n'ont-elles pas changé grand-chose?",
    },
    "cps21_own_fin_future": {
        "en": "Over the next year, do you think your financial situation will get better, stay about the same, or get worse?",
        "fr": "Au cours de la prochaine année, pensez-vous que votre situation financière va s'améliorer, rester à peu près la même, ou se détériorer?",
    },
    "cps21_imm": {
        "en": "Do you think Canada should admit more immigrants, fewer immigrants, or about the same number of immigrants as now?",
        "fr": "Pensez-vous que le Canada devrait admettre plus d'immigrants, moins d'immigrants, ou à peu près le même nombre qu'actuellement?",
    },
    "cps21_refugees": {
        "en": "Do you think Canada should admit more refugees, fewer refugees, or about the same number of refugees as now?",
        "fr": "Pensez-vous que le Canada devrait admettre plus de réfugiés, moins de réfugiés, ou à peu près le même nombre qu'actuellement?",
    },
    "cps21_candidate_imag": {
        "en": "Imagine you were the only voter in the election. Which candidate would you want to win in your riding?",
        "fr": "Imaginez que vous êtes le seul électeur ou la seule électrice dans votre circonscription. Quel candidat ou quelle candidate aimeriez-vous voir gagner?",
    },
    "cps21_comfort3": {
        "en": "Regardless of how you voted, how comfortable were you with voting in person during the coronavirus (COVID-19) pandemic?",
        "fr": "Quelle que soit la façon dont vous avez voté, dans quelle mesure étiez-vous à l'aise avec l'idée de voter en personne pendant la pandémie de coronavirus (COVID-19)?",
    },
    "cps21_covid_liberty": {
        "en": "Public health recommendations aimed at slowing COVID-19 spread are threatening my liberty.",
        "fr": "Les recommandations de la santé publique visant à ralentir la propagation de la COVID-19 menacent ma liberté.",
    },
    "cps21_debate_fr": {
        "fr": "Avez-vous écouté ou regardé le débat des chefs fédéraux en français le jeudi 2 septembre?",
    },
    "cps21_debate_fr2": {
        "fr": "Et le débat des chefs fédéraux en français le mercredi 8 septembre? L'avez-vous regardé ou écouté?",
    },
    "cps21_duty_choice": {
        "fr": "Pour vous personnellement, est-ce que voter est d'abord un devoir ou un choix?",
    },
    "cps21_govt_confusing": {
        "en": "Sometimes, politics and government seem so complicated that a person like me can't really understand what's going on.",
        "fr": "Parfois, la politique et le gouvernement semblent si compliqués qu'une personne comme moi ne peut pas vraiment comprendre ce qui se passe.",
    },
    "cps21_govt_say": {
        "fr": "Les gens comme moi n'ont rien à dire sur ce que le gouvernement fait.",
    },
    "cps21_imp_iss": {
        "fr": "Pour vous personnellement, quel est l'enjeu le plus important de cette élection fédérale?",
    },
    "cps21_lr_scale_bef_1": {
        "fr": "En politique, on parle parfois de gauche et de droite. Où vous placeriez-vous sur cette échelle?",
    },
    "cps21_news_cons": {
        "fr": "En moyenne, combien de temps passez-vous chaque jour à lire, regarder et écouter les nouvelles?",
    },
    "cps21_pol_eth": {
        "fr": "Il est important que les politiciens se comportent de façon éthique dans l'exercice de leurs fonctions.",
    },
    "cps21_pos_carbon": {
        "en": "To help reduce greenhouse gas emissions, the federal government should continue the carbon tax.",
        "fr": "Afin d'aider à réduire les émissions de gaz à effet de serre, le gouvernement fédéral devrait maintenir la taxe sur le carbone.",
    },
    "cps21_pos_envreg": {
        "en": "Environmental regulation should be stricter, even if it means consumers pay higher prices.",
        "fr": "La réglementation environnementale devrait être plus stricte, même si elle oblige les consommateurs à payer des prix plus élevés.",
    },
    "cps21_pos_jobs": {
        "en": "When there is a conflict between protecting the environment and creating jobs, jobs should come first.",
        "fr": "Lorsqu'il existe un conflit entre la protection de l'environnement et la création d'emplois, les emplois devraient avoir la priorité.",
    },
    "cps21_pos_life": {
        "en": "Individuals who are terminally ill should be allowed to end their lives with the assistance of a doctor.",
        "fr": "Les personnes en phase terminale devraient pouvoir mettre fin à leurs jours avec l'aide d'un médecin.",
    },
    "cps21_pos_trade": {
        "en": "There should be more free trade with other countries, even if it hurts some industries in Canada.",
        "fr": "Il devrait y avoir plus de libre-échange avec d'autres pays, même si cela nuit à certaines industries au Canada.",
    },
    "cps21_premier_name": {
        "fr": "Sans vérifier en ligne, vous souvenez-vous du nom du premier ministre ou de la première ministre de votre province?",
    },
    "cps21_spoil": {
        "fr": "Avez-vous déjà intentionnellement annulé votre vote lors d'une élection (c'est-à-dire rempli votre bulletin de façon à ce qu'il ne soit compté pour aucun candidat)?",
    },
    "cps21_volunteer": {
        "en": "In the past 12 months, how many times did you volunteer for a group or organization such as a school, a religious organization, or sports/community associations?",
        "fr": "Au cours des 12 derniers mois, combien de fois avez-vous fait du bénévolat pour un groupe ou un organisme comme une école, une organisation religieuse ou une association sportive ou communautaire?",
    },
    "cps21_debate_en": {
        "en": "Did you happen to watch or listen to the English-language federal leaders' debate on Wednesday, September 9th?",
        "fr": "Avez-vous écouté ou regardé le débat des chefs fédéraux en anglais le lundi 7 octobre?",
    },
    "cps21_demsat": {
        "fr": "Dans l'ensemble, quel est votre niveau de satisfaction quant au fonctionnement de la démocratie au Canada?",
    },
    "cps21_pos_fptp": {
        "en": "Canada should change its electoral system from First Past the Post to a proportional representation system.",
        "fr": "Le Canada devrait changer son système électoral afin de passer du mode de scrutin actuel à un système de représentation proportionnelle.",
    },
    "cps21_pos_energy": {
        "en": "The federal government should do more to help Canada's energy sector, including building oil pipelines.",
        "fr": "Le gouvernement fédéral devrait en faire davantage afin d'aider le secteur énergétique canadien, notamment en construisant des oléoducs.",
    },
    "cps21_pos_subsid": {
        "en": "The federal government should end all corporate and economic development subsidies.",
        "fr": "Le gouvernement fédéral devrait mettre fin à toutes les subventions au développement des entreprises et de l'économie.",
    },
    "cps21_residential_2a": {
        "en": "Accelerate progress on the calls to action from the Truth and Reconciliation Commission.",
        "fr": "Accélérer la mise en œuvre des appels à l'action lancés par la Commission de vérité et de réconciliation.",
    },
    "cps21_residential_2d": {
        "en": "Renaming buildings and institutions that are named for people who built or ran parts of the residential school system.",
        "fr": "Renommer les bâtiments et les institutions qui portent le nom de personnes qui ont construit ou dirigé en partie le système de pensionnats.",
    },
    "cps21_comfort2": {
        "en": "If you decide to vote, how comfortable are you with the idea of voting in person during the coronavirus (COVID-19) pandemic?",
        "fr": "Si vous décidez de voter, à quel point êtes-vous à l'aise de voter en personne pendant la pandémie de coronavirus (COVID-19)?",
    },
    "cps21_camp_issue": {
        "en": "Thinking about what politicians and the media have been discussing during this election campaign, what issue would you say the campaign has focused on the most?",
        "fr": "Si vous réfléchissez à ce dont les politiciens et les médias parlent pendant cette campagne électorale, quel est selon vous l'enjeu sur lequel la campagne porte le plus?",
    },
}


def split_parent_child_label(label: str) -> tuple[str, str | None]:
    """Split labels like 'Parent - Child' into (parent, child)."""
    text = (label or "").strip()
    if not text:
        return "", None
    if " - " in text:
        parent, child = text.rsplit(" - ", 1)
        parent = parent.strip()
        child = child.strip()
        if parent and child:
            return parent, child
    return text, None


def extract_child_label_from_parent_question(question: str, var_name: str) -> str | None:
    """Extract child label from parent question text when variable marker exists.

    Example expected fragment:
        ... ▢ Prestation ... (cps21_covidrelief__1) ...
    """
    q = (question or "").strip()
    if not q:
        return None

    marker = f"({var_name})"
    idx = q.find(marker)
    if idx == -1:
        return None

    prefix = q[:idx]
    for sep in ("▢", "□", "■", ";"):
        if sep in prefix:
            prefix = prefix.rsplit(sep, 1)[-1]

    candidate = prefix.strip(" -:\t")
    if len(candidate) < 3:
        return None
    return candidate


def clean_codebook_text(text: str) -> str:
    """Remove Qualtrics artifacts and normalize whitespace."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"Display This (Question|Choice):", "", cleaned)
    cleaned = re.sub(r"\$\{[^}]+\}", "", cleaned)
    cleaned = re.sub(r"\[[^\]]+\]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip(" -")


def load_stata_data(path: str) -> tuple[pd.DataFrame, dict, dict]:
    """Load Stata file with value labels."""
    logger.info(f"Loading {path}...")
    reader = StataReader(path, convert_categoricals=False)
    df = reader.read()
    var_labels = reader.variable_labels()
    value_labels = reader.value_labels()
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    # Fix encoding: Stata files may have latin-1 encoded strings
    for col in df.select_dtypes(include=["object", "string"]).columns:
        try:
            df[col] = df[col].apply(
                lambda x: x.encode("latin-1").decode("utf-8", errors="ignore")
                if isinstance(x, str) else x
            )
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

    # Fix value_labels encoding (latin-1 → UTF-8)
    for var_name, labels_dict in value_labels.items():
        try:
            fixed = {}
            for key, lstr in labels_dict.items():
                if isinstance(lstr, str):
                    try:
                        fixed[key] = lstr.encode("latin-1").decode("utf-8", errors="ignore")
                    except (UnicodeDecodeError, UnicodeEncodeError):
                        fixed[key] = lstr
                else:
                    fixed[key] = lstr
            value_labels[var_name] = fixed
        except Exception:
            pass

    return df, var_labels, value_labels


def load_codebook_jsonl(path: str) -> tuple[dict, dict]:
    """Load question metadata from JSONL codebook, returning EN and FR dicts separately.

    The JSONL has two entries per variable: English first, French second.
    Heuristic: an entry is French if its label or question contains accented characters
    or if it's the second entry seen for that variable.
    """
    logger.info(f"Loading codebook from {path}...")
    codebook_en: dict = {}
    codebook_fr: dict = {}
    seen: dict = {}  # var -> first entry already stored
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            var = item.get("variable")
            if not var:
                continue
            if var not in seen:
                seen[var] = item
                codebook_en[var] = item
            else:
                # Second entry for this variable → French version
                codebook_fr[var] = item
    logger.info(
        f"Loaded {len(codebook_en)} EN + {len(codebook_fr)} FR question entries"
    )
    return codebook_en, codebook_fr


def load_thematic_domains(path: str) -> dict:
    """Load thematic domain mapping from JSON file."""
    logger.info(f"Loading thematic domains from {path}...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} thematic domain mappings")
    return data


def filter_quebec(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to Quebec respondents only (cps21_province == 11)."""
    initial = len(df)
    df_qc = df[df["cps21_province"] == 11].copy()
    logger.info(
        f"Filtered to Quebec: {len(df_qc)} of {initial} "
        f"({100 * len(df_qc) / initial:.1f}%)"
    )
    return df_qc


def select_relevant_variables(
    df: pd.DataFrame, codebook: dict, thematic_domains: dict
) -> list[str]:
    """
    Select relevant variables for the analysis.

    Categories:
    - SES: age (yob), province, education, gender, language
    - Attitude questions: only variables explicitly classified in thematic_domains.json

    This keeps the analysis universe fixed and avoids unclassified variables leaking
    into the train split by default.
    """
    ses_vars = [
        "cps21_yob",        # Year of birth (converted to age)
        "cps21_province",
        "cps21_education",
        "cps21_genderid",
        "cps21_language_1",
    ]
    selected = [v for v in ses_vars if v in df.columns]
    selected_set = set(selected)

    # Always include all variables explicitly classified in thematic_domains.json
    for var in thematic_domains:
        if var in df.columns and var not in selected_set:
            selected.append(var)
            selected_set.add(var)

    logger.info(
        "Selected %d variables (%d SES + %d thematic variables)",
        len(selected),
        len(ses_vars),
        len(selected) - len(ses_vars),
    )
    return selected


def apply_value_labels(df: pd.DataFrame, value_labels: dict, var: str) -> pd.Series:
    """Convert numeric column to text labels. Unmapped values stay as strings."""
    if var not in value_labels:
        return df[var].astype(str)
    labels = value_labels[var]
    # Some DTA label sets store values in a different case than the codebook
    # options used to build the en_to_fr localization map in downstream scripts.
    # Apply explicit normalization mappings where needed.
    if var in VALUE_LABEL_NORMALIZATIONS:
        norm = VALUE_LABEL_NORMALIZATIONS[var]
        labels = {k: norm.get(v.lower(), v) for k, v in labels.items()}
    return df[var].map(labels).fillna(df[var].astype(str))


def compute_age(df: pd.DataFrame) -> pd.Series:
    """Compute age from year of birth. Survey conducted in 2021."""
    yob = pd.to_numeric(df["cps21_yob"], errors="coerce").astype("Int64")
    return (2021 - yob).astype(str)


def compute_language_profile(df: pd.DataFrame, value_labels: dict) -> pd.Series:
    """Collapse language dummies into 3 categories: English/French/Other.

    Classification rules:
    1. Purely English (only EN checked): → "English"
    2. Purely French (only FR checked): → "French"
    3. Has EN or FR, but also checked other languages (e.g. EN+FR bilinguals,
       FR+Italian, EN+Spanish, EN+FR+Arabic, ...): fall back to survey interface
       language (UserLanguage: FR-CA → "French", EN → "English"). These respondents
       clearly understand at least one official language; their chosen survey language
       is the best proxy for their dominant official language.
    4. Other language only (no EN, no FR checked): → "Other"
    5. Nothing checked: → "Not specified"
    """

    available = [col for col in LANGUAGE_DUMMY_COLS if col in df.columns]
    if not available:
        return pd.Series(["Not specified"] * len(df), index=df.index)

    cols_for_work = list(available)
    if "UserLanguage" in df.columns:
        cols_for_work.append("UserLanguage")
    work_df = df[cols_for_work]

    def row_to_language(row: pd.Series) -> str:
        english = bool(pd.notna(row.get("cps21_language_1")))
        french = bool(pd.notna(row.get("cps21_language_2")))
        other = any(
            pd.notna(row.get(col))
            for col in available
            if col not in {"cps21_language_1", "cps21_language_2"}
        )

        # Unambiguous single-language cases
        if english and not french and not other:
            return "English"
        if french and not english and not other:
            return "French"

        # Has at least one official language (EN or FR) — possibly also others:
        # use survey interface language as tiebreaker.
        if english or french:
            ul = str(row.get("UserLanguage", "EN"))
            return "French" if ul == "FR-CA" else "English"

        # No EN or FR at all — genuinely allophone respondent.
        if other:
            return "Other"

        return "Not specified"

    return work_df.apply(row_to_language, axis=1)


def compute_voted_2019(df: pd.DataFrame, value_labels: dict) -> pd.Series:
    """Compute binary voted_2019 from cps21_vote_2019."""
    # Convert 'cps21_vote_2019' to its text labels first
    vote_2019_series = apply_value_labels(df, value_labels, "cps21_vote_2019")
    # 'yes' if not null and not 'Did not vote', else 'no'
    return vote_2019_series.apply(
        lambda x: "yes" if pd.notna(x) and x != "Did not vote" else "no"
    )


def create_respondent_split(respondents: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42) -> pd.Series:
    """Create a stratified train/test split by language (FR/EN).
    
    This split is used only for condition 4B to test generalization to new respondents.
    The stratification ensures both language groups are represented proportionally in 
    both train and test sets.
    
    Args:
        respondents: DataFrame with 'language' column
        test_size: Fraction of data for test set (default 0.2 for 80/20 split)
        random_seed: Random seed for reproducibility (default 42)
    
    Returns:
        Series with 'train' or 'test' values
    """
    from sklearn.model_selection import train_test_split
    
    # Create a simplified language column for stratification: group Other as non-French
    lang_for_split = respondents["language"].apply(
        lambda x: "FR" if x == "French" else "non-FR"
    ).values
    
    # Generate row positions (0 to n-1)
    n_respondents = len(respondents)
    all_positions = np.arange(n_respondents)
    
    # Use sklearn's stratified split with positions
    train_positions, test_positions = train_test_split(
        all_positions,
        test_size=test_size,
        random_state=random_seed,
        stratify=lang_for_split
    )
    
    # Create split series using position-based indexing
    split_series = pd.Series("train", index=respondents.index)
    split_series.iloc[test_positions] = "test"
    
    # Log split statistics
    logger.info(f"Respondent split (condition 4B):")
    logger.info(f"  Total: {len(respondents)}")
    logger.info(f"  Train: {(split_series == 'train').sum()} ({100 * (split_series == 'train').sum() / len(respondents):.1f}%)")
    logger.info(f"  Test: {(split_series == 'test').sum()} ({100 * (split_series == 'test').sum() / len(respondents):.1f}%)")
    
    # Log stratification by language
    for lang in ["French", "English", "Other"]:
        lang_mask = respondents["language"] == lang
        if lang_mask.sum() > 0:
            train_count = (split_series[lang_mask] == "train").sum()
            test_count = (split_series[lang_mask] == "test").sum()
            total = lang_mask.sum()
            logger.info(f"  {lang}: {total} total (train={train_count}, test={test_count})")
    
    return split_series


def prepare_respondents(
    df: pd.DataFrame, selected_vars: list[str], value_labels: dict
) -> pd.DataFrame:
    """Build respondents matrix with text labels."""
    logger.info("Transforming values to text labels...")
    respondents = pd.DataFrame(index=df.index)

    rename = {
        "cps21_yob": ("age", lambda: compute_age(df)),
        "cps21_language_1": ("language", lambda: compute_language_profile(df, value_labels)),
        "cps21_province": ("province", lambda: apply_value_labels(df, value_labels, "cps21_province")),
        "cps21_education": ("education", lambda: apply_value_labels(df, value_labels, "cps21_education")),
        "cps21_genderid": ("gender", lambda: apply_value_labels(df, value_labels, "cps21_genderid")),
        "cps21_vote_2019": ("voted_2019", lambda: compute_voted_2019(df, value_labels)),
    }

    # Survey interface language (FR-CA or EN) — used to select prompt language
    if "UserLanguage" in df.columns:
        respondents["survey_language"] = df["UserLanguage"].astype(str)
    else:
        respondents["survey_language"] = "EN"

    for var in selected_vars:
        if var in rename:
            col_name, fn = rename[var]
            respondents[col_name] = fn()
        else:
            respondents[var] = apply_value_labels(df, value_labels, var)

    logger.info(f"Respondents matrix: {len(respondents)} × {len(respondents.columns)}")
    return respondents


TEST_DOMAINS = {"vote_choice", "national_identity"}
DROP_DOMAINS = {"_drop"}


def get_split(domain: str | None) -> str:
    """Map a thematic domain to train/test/drop."""
    if domain is None:
        return "drop"
    if domain in DROP_DOMAINS:
        return "drop"
    if domain in TEST_DOMAINS:
        return "test"
    return "train"


def prepare_questions(
    selected_vars: list[str],
    var_labels: dict,
    value_labels: dict,
    codebook_en: dict,
    codebook_fr: dict,
    thematic_domains: dict,
) -> pd.DataFrame:
    """Build questions metadata table with thematic domain, split, and FR columns."""
    logger.info("Preparing questions metadata...")

    col_rename = {
        "cps21_language_1": "language",
        "cps21_province": "province",
        "cps21_education": "education",
        "cps21_genderid": "gender",
    }

    rows = []
    for var in selected_vars:
        if var == "cps21_yob":
            continue  # replaced by age, not an attitude question

        label = var_labels.get(var, "")
        question_text = codebook_en.get(var, {}).get("question", "")

        options = codebook_en.get(var, {}).get("options", [])
        if not options and var in value_labels:
            options = [
                f"{k}: {v}" for k, v in sorted(value_labels[var].items())
            ]

        # French versions — fallback to parent variable (strip trailing _N suffix)
        # when the exact variable is absent (e.g. cps21_not_vote_for_1 → cps21_not_vote_for)
        parent_var = re.sub(r"_+\d+$", "", var)
        en_entry = codebook_en.get(var) or codebook_en.get(parent_var) or {}
        fr_entry_exact = codebook_fr.get(var)
        fr_entry_parent = codebook_fr.get(parent_var)
        fr_entry = fr_entry_exact or fr_entry_parent or {}

        label_fr = fr_entry.get("label", "")
        question_fr = fr_entry.get("question", "")
        options_fr = fr_entry.get("options", [])

        label = clean_codebook_text(label)
        question_text = clean_codebook_text(question_text)
        label_fr = clean_codebook_text(label_fr)
        question_fr = clean_codebook_text(question_fr)

        is_child = bool(re.search(r"_+\d+$", var))
        if is_child:
            parent_label_en, child_label_en = split_parent_child_label(label)
            parent_label_fr, child_label_fr = split_parent_child_label(label_fr)

            manual_parent = MANUAL_PARENT_LABELS.get(parent_var, {})
            if manual_parent.get("en"):
                parent_label_en = manual_parent["en"]
            if manual_parent.get("fr"):
                parent_label_fr = manual_parent["fr"]

            # If FR child is missing (common when only parent entry exists), try
            # extracting from parent question text with explicit (variable_name) markers.
            if not child_label_fr and fr_entry_parent:
                child_label_fr = extract_child_label_from_parent_question(
                    fr_entry_parent.get("question", ""),
                    var,
                )

            # Mirror behavior for EN when codebook has only parent text.
            if not child_label_en:
                child_label_en = extract_child_label_from_parent_question(
                    en_entry.get("question", ""),
                    var,
                )

            manual_child = MANUAL_CHILD_LABELS.get(var, {})
            if manual_child.get("en"):
                child_label_en = manual_child["en"]
            if manual_child.get("fr"):
                child_label_fr = manual_child["fr"]

            if parent_label_en and child_label_en:
                label = f"{parent_label_en} - {child_label_en}"
            if parent_label_fr and child_label_fr:
                label_fr = f"{parent_label_fr} - {child_label_fr}"

            # Parent-level options are often unusable for child variables (e.g. "e (6)").
            # Keep child options only when exact variable-level FR entry exists.
            if fr_entry_exact is None:
                options_fr = []
            # For certain battery variables the codebook has no FR options at all;
            # use the manually specified list keyed by parent variable name.
            if not options_fr and parent_var in MANUAL_OPTIONS_FR:
                options_fr = MANUAL_OPTIONS_FR[parent_var]

        manual_override = MANUAL_LABEL_OVERRIDES.get(var, {})
        if manual_override.get("en"):
            label = manual_override["en"]
        if manual_override.get("fr"):
            label_fr = manual_override["fr"]

        domain_entry = thematic_domains.get(var, {})
        domain = domain_entry.get("domain") if isinstance(domain_entry, dict) else None
        split = get_split(domain)

        rows.append({
            "variable_name": var,
            "column_name": col_rename.get(var, var),
            "label": label,
            "question": question_text,
            "options": json.dumps(options, ensure_ascii=False) if options else "",
            "label_fr": label_fr,
            "question_fr": question_fr,
            "options_fr": json.dumps(options_fr, ensure_ascii=False) if options_fr else "",
            "thematic_domain": domain,
            "split": split,
        })

    df_q = pd.DataFrame(rows)

    counts = df_q["split"].value_counts()
    logger.info(
        f"Questions table: {len(df_q)} rows "
        f"(train={counts.get('train', 0)}, "
        f"test={counts.get('test', 0)}, "
        f"drop={counts.get('drop', 0)})"
    )
    return df_q


def main() -> None:
    data_dir = Path("data")
    src_dir = Path("src/article_silicon_sampling_quebec")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df, var_labels, value_labels = load_stata_data(
        str(data_dir / "raw" / "ces_2021" / "ces_2021.dta")
    )
    codebook_en, codebook_fr = load_codebook_jsonl(
        str(data_dir / "raw" / "ces_2021" / "ces_2021_codebook_questions.jsonl")
    )
    thematic_domains = load_thematic_domains(
        str(src_dir / "thematic_domains.json")
    )

    df_qc = filter_quebec(df)
    selected_vars = select_relevant_variables(df_qc, codebook_en, thematic_domains)
    respondents = prepare_respondents(df_qc, selected_vars, value_labels)
    questions = prepare_questions(
        selected_vars, var_labels, value_labels,
        codebook_en, codebook_fr, thematic_domains,
    )

    # Exclude _drop variables from respondents matrix
    keep_cols = questions.loc[questions["split"] != "drop", "column_name"].tolist()
    # Always keep SES columns and survey metadata (no split assignment)
    ses_cols = ["survey_language", "age", "province", "education", "gender", "language", "voted_2019"]
    keep_cols = [c for c in respondents.columns if c in ses_cols or c in keep_cols]
    respondents = respondents[keep_cols]
    logger.info(f"Respondents matrix after dropping _drop vars: {len(respondents)} × {len(respondents.columns)}")

    # Add respondent_split column for condition 4B (train/test split for generalization to new respondents)
    respondents["respondent_split"] = create_respondent_split(respondents)
    logger.info(f"Respondents matrix after adding respondent_split: {len(respondents)} × {len(respondents.columns)}")

    # Save
    pl.from_pandas(questions).write_parquet(str(processed_dir / "questions.parquet"))
    questions.to_csv(str(processed_dir / "questions.csv"), index=False, encoding="utf-8")
    logger.info(f"Saved questions → {processed_dir / 'questions.parquet'}")

    pl.from_pandas(respondents).write_parquet(str(processed_dir / "respondents.parquet"))
    respondents.to_csv(str(processed_dir / "respondents.csv"), index=False, encoding="utf-8")
    logger.info(f"Saved respondents → {processed_dir / 'respondents.parquet'}")

    logger.info("✓ Data preparation complete")


if __name__ == "__main__":
    main()
