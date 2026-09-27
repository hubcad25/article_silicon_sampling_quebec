# CES 2021 Codebook (PDF -> Markdown)

- Source PDF: `data/raw/ces_2021/ces_2021_codebook.pdf`
- Generated (UTC): `2026-03-26T02:09:10+00:00`
- Variables extracted: `558` (`CPS=318`, `PES=240`)

## Campaign Period Survey (CPS)

### `cps21_consent_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_consent`
- Label: Letter of Information and Consent
- Question: Ontario, (519) 661-2111 x85164, laura.stephenson@uwo.ca Montréal, (514) 987-3000 x5676, harell.allison@uqam.ca Peter Loewen, PhD, Political Science, Munk School of Global Affairs, University of Toronto, (416) 978-5120, peter.loewen@utoronto.ca
- Options:
  - I consent to participate in this study. I have read all of the information about the study. (1)
  - I do not consent to participate. (2)

### `cps21_consent`
- Label: Lettre d'information et de consentement
- Question: Titre du projet: Étude électorale canadienne 2021 Chercheure principale: Laura Stephenson, PhD, Science politique, University of Western Ontario, (519) 661-2111 x 85164, laura.stephenson@uwo.ca Co-chercheurs: Allison Harell, PhD, Science politique, Université du Québec à Montréal,
- Options:
  - Daniel Rubenson, PhD, Politiques et administration publique, Ryerson University, (416)
  - Je consens à participer à cette étude. J'ai lu toutes les informations sur l'étude. (1)
  - Je ne consens pas à participer. (2)

### `cps21_captcha_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_captcha_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_captcha`
- Label: Before you proceed to the survey, please complete the Captcha below.

### `cps21_captcha`
- Label: Avant d’accéder au sondage, veuillez compléter le Captcha ci-dessous.
- Question: Demographics

### `cps21_citizenship`
- Label: Are you a...
- Options:
  - Canadian citizen (1)
  - Permanent resident (2)
  - Other (3)

### `cps21_citizenship`
- Label: Êtes-vous...
- Options:
  - Citoyen(ne) canadien(ne) (1)
  - Résident(e) permanent(e) (2)
  - Autre (3)

### `cps21_yob`
- Label: To make sure we are talking to a cross section of Canadians, we need to
- Question: get a little information about your background. First, in what year were you born?
- Options:
  - ▼ 1920 (1) ... 2010 (91)

### `cps21_yob`
- Label: Afin d’être certains que nous nous adressons à un échantillon représentatif
- Question: des Canadiens, nous avons besoin d’informations de base sur vous. Tout d'abord, en quelle année êtes-vous né(e)? Respondents who were born in 2003 received an additional question to determine whether they were 18 years of age or not.
- Options:
  - ▼ 1920 (1) ... 2010 (91)

### `cps21_yob_2003_age`
- Label: How old are you?
- Options:
  - 17 (1)
  - 18 (2)

### `cps21_yob_2003_age`
- Label: Quel âge avez-vous?
- Options:
  - 17 (1)
  - 18 (2)

### `cps21_genderid`
- Label: Are you...?
- Options:
  - A man (1)
  - A woman (2)
  - Non-binary (3)
  - Another gender, please specify (4): cps21_genderid_4_TEXT

### `cps21_genderid`
- Label: Êtes-vous...
- Options:
  - Un homme (1)
  - Une femme (2)
  - Non-binaire (3)
  - Autre genre, veuillez spécifier (4): cps21_genderid_4_TEXT

### `cps21_trans`
- Label: Are you transgender?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to say (3)

### `cps21_trans`
- Label: Êtes-vous une personne trans?
- Options:
  - Oui (1)
  - Non (2)
  - Ne sais pas/Préfère ne pas répondre (3)

### `cps21_province`
- Label: In which province or territory are you currently living?
- Options:
  - Alberta (1)
  - British Columbia (2)
  - Manitoba (3)
  - New Brunswick (4)
  - Newfoundland and Labrador (5)
  - Northwest Territories (6)
  - Nova Scotia (7)
  - Nunavut (8)
  - Ontario (9)
  - Prince Edward Island (10)
  - Quebec (11)
  - Saskatchewan (12)
  - Yukon (13)

### `cps21_province`
- Label: Dans quelle province ou territoire habitez-vous présentement?
- Options:
  - Alberta (1)
  - Colombie-Britannique (2)
  - Manitoba (3)
  - Nouveau-Brunswick (4)
  - Terre-Neuve-et-Labrador (5)
  - Territoires du Nord-Ouest (6)
  - Nouvelle-Écosse (7)
  - Nunavut (8)
  - Ontario (9)
  - Île-du-Prince-Édouard (10)
  - Québec (11)
  - Saskatchewan (12)
  - Yukon (13)

### `cps21_postalcode`
- Label: Please enter your six-digit postal code in the box below. (For
- Question: example "A1A 1A1", with letters in uppercase) We are collecting your postal code in order to compare our results to census and electoral district data. Your postal code will not be released publicly or shared with any third party.

### `cps21_postalcode`
- Label: Veuillez inscrire votre code postal à six chiffres dans la case ci-
- Question: dessous. (Par exemple, "A1A 1A1", avec des lettres majuscules) Nous recueillons votre code postal afin de comparer nos résultats aux données de recensement et de circonscription. Votre code postal ne sera pas divulgué publiquement ni partagé avec des tiers.

### `cps21_education`
- Label: What is the highest level of education that you have completed?
- Options:
  - No schooling (1)
  - Some elementary school (2)
  - Completed elementary school (3)
  - Some secondary/ high school (4)
  - Completed secondary/ high school (5)
  - Some technical, community college, CEGEP, College Classique (6)
  - Completed technical, community college, CEGEP, College Classique (7)
  - Some university (8)
  - Bachelor's degree (9)
  - Master's degree (10)
  - Professional degree or doctorate (11)
  - Don't know/ Prefer not to answer (12)

### `cps21_education`
- Label: Quel est votre plus haut niveau de scolarité complété?
- Question: Satisfaction with democracy
- Options:
  - Aucune scolarité (1)
  - Quelques années d'école primaire (2)
  - École primaire terminée (3)
  - Quelques années d'école secondaire (4)
  - École secondaire terminée (5)
  - Quelques années d'études au collègue, au cégep ou au collège classique (6)
  - Études terminées au collège, au cégep, ou au collège classique (7)
  - Quelques années d'études universitaires (8)
  - Baccalauréat (9)
  - Maîtrise (10)
  - Diplôme professionnel ou doctorat (11)
  - Ne sais pas/Préfère ne pas répondre (12)

### `cps21_demsat`
- Label: On the whole, how satisfied are you with the way democracy works in
- Question: Canada?
- Options:
  - Very satisfied (1)
  - Fairly satisfied (2)
  - Not very satisfied (3)
  - Not at all satisfied (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_demsat`
- Label: Dans l'ensemble, quel est votre niveau de satisfaction quant au
- Question: fonctionnement de la démocratie au Canada?
- Options:
  - Très satisfait (1)
  - Assez satisfait (2)
  - Pas très satisfait (3)
  - Pas du tout satisfait (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_imp_iss_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_imp_iss_t`
- Label: Timing
- Question: Most important issue
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_imp_iss`
- Label: What is the most important issue to you personally in this federal
- Question: election? ________________________________________________________________

### `cps21_imp_iss`
- Label: Pour vous personnellement, quel est l'enjeu le plus important de cette
- Question: élection fédérale? ________________________________________________________________ Display This Question: Response Is Not Empty

### `cps21_imp_iss_party`
- Label: Which party is best at addressing this issue?
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_imp_iss_party_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_imp_iss_party`
- Label: Quel parti aborde le mieux cet enjeu?
- Question: each response option in cps21_imp_iss_party.
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_imp_iss_party_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_imp_loc_iss_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_imp_loc_iss_t`
- Label: Timing
- Question: Display This Question:
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_imp_loc_iss`
- Label: What is the most important local issue to you personally in this
- Question: federal election? ________________________________________________________________

### `cps21_imp_loc_iss`
- Label: Pour vous personnellement, quel est l'enjeu local le plus
- Question: important de cette élection fédérale? ________________________________________________________________ Display This Question: Text Response Is Not Empty

### `cps21_imp_loc_iss_p`
- Label: Which party is best at addressing this local issue?
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_imp_loc_iss_p_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_imp_loc_iss_p`
- Label: Quel parti aborde le mieux cet enjeu local?
- Question: Campaign Issues
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_imp_loc_iss_p_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_camp_issue_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_camp_issue_t`
- Label: Timing
- Question: Display This Question:
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_camp_issue`
- Label: Thinking about what politicians and the media have been
- Question: discussing during this election campaign, what issue would you say the election campaign has been focused on the most? ________________________________________________________________

### `cps21_camp_issue`
- Label: Si vous réfléchissez à ce dont les politiciens et les médias parlent
- Question: pendant cette campagne électorale, quel est, selon vous, l’enjeu sur lequel la campagne électorale porte le plus? ________________________________________________________________ Political interest

### `cps21_interest_gen_1`
- Label: How interested are you in politics generally? Set the slider to
- Question: a number from 0 to 10, where 0 means no interest at all, and 10 means a great deal of interest. No interest at A great deal of Don't know/ all interest Prefer not to

### `cps21_interest_gen_1`
- Label: Quel est votre niveau d'intérêt pour la politique en général?
- Question: Veuillez glisser le curseur sur un chiffre de 0 à 10, où 0 indique aucun intérêt du tout et 10 indique beaucoup d'intérêt. Aucun intérêt Beaucoup Je ne sais d'intérêt pas/Préfère ne

### `cps21_interest_elxn_1`
- Label: How interested are you in this federal election? Set the slider
- Question: to a number from 0 to 10, where 0 means no interest at all, and 10 means a great deal of interest. No interest at A great deal of Don't know/ all interest Prefer not to

### `cps21_interest_elxn_1`
- Label: Quel est votre niveau d'intérêt pour cette élection fédérale?
- Question: Veuillez glisser le curseur sur un chiffre de 0 à 10, où 0 indique aucun intérêt du tout et 10 indique beaucoup d'intérêt. Aucun intérêt Beaucoup Je ne sais d'intérêt pas/Préfère ne

### `cps21_v_likely`
- Label: On election day, are you...
- Options:
  - Certain to vote (1)
  - Likely to vote (2)
  - Unlikely to vote (3)
  - Certain not to vote (4)
  - I am not eligible to vote (5)
  - I already voted (by mail, advance poll, etc.) (6)
  - Don't know/ Prefer not to answer (7)

### `cps21_v_likely`
- Label: Lors du jour de l'élection, est-il...
- Question: Display This Question:
- Options:
  - Certain que vous votiez (1)
  - Probable que vous votiez (2)
  - Improbable que vous votiez (3)
  - Certain que vous ne votiez pas (4)
  - Je ne suis pas éligible au vote (5)
  - J'ai déjà voté (par le poste, par anticipation, etc.) (6)
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_v_likely_pr`
- Label: If you become a Canadian citizen, how likely are you to vote in the
- Question: first election for which you are eligible?
- Options:
  - Certain to vote (1)
  - Likely to vote (2)
  - Unlikely to vote (3)
  - Certain not to vote (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_v_likely_pr`
- Label: Si vous devenez citoyen(ne) canadien(ne), quelle est la probabilité
- Question: que vous votiez à la première élection à laquelle vous êtes admissible? Display This Question: Or On election day, are you... = Likely to vote And If
- Options:
  - Certain(e) de voter (1)
  - Probable de voter (2)
  - Improbable de voter (3)
  - Certain(e) de ne pas voter (4)
  - Je ne sais pas/Préfère ne pas répondre (5)
  - In person, on Election Day (1)
  - In person, at an advance poll (2)
  - In person, at an Elections Canada office (3)
  - By mail (4)
  - Not sure (5)
  - Prefer not to answer (6)
  - En personne, le jour de l'élection (1)
  - En personne, par anticipation (2)
  - En personne, dans un bureau d’Élections Canada (3)
  - Par la poste (4)
  - Je ne suis pas certain(e) (5)
  - Préfère ne pas répondre (6)

### `cps21_howvote2`
- Label: How did you vote?
- Options:
  - In person, at an advance poll (1)
  - In person, at an Elections Canada office (2)
  - By mail (3)
  - Prefer not to answer (4)

### `cps21_howvote2`
- Label: Quelle méthode avez-vous utilisée pour voter?
- Question: Display This Question: election, what voting method do you think you would use? (Select one) méthode de vote pensez-vous utiliser? (sélectionnez un choix) Comfortable to vote in person during the pandemic
- Options:
  - En personne, par anticipation (1)
  - En personne, dans un bureau d’Élections Canada (2)
  - Par la poste (3)
  - Préfère ne pas répondre (4)
  - In person, on Election Day (1)
  - In person, at an advance poll (2)
  - In person, at an Elections Canada office (3)
  - By mail (4)
  - Not sure (5)
  - Prefer not to answer (6)
  - En persone, par anticipation (1)
  - En personne, le jour du scrutin (2)
  - En personne, dans un bureau d’Élections Canada (3)
  - Par la poste (4)
  - Je ne suis pas certain(e) (5)
  - Préfère ne pas répondre (6)

### `cps21_comfort1`
- Label: Regardless of how you plan to vote, how comfortable are you with the
- Question: idea of voting in person during the coronavirus (COVID-19) pandemic?
- Options:
  - Very comfortable (1)
  - Somewhat comfortable (2)
  - Not very comfortable (3)
  - Not comfortable at all (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_comfort1`
- Label: Quelle que soit la façon dont vous comptez voter, êtes-vous à l'aise
- Question: avec l'idée de voter en personne pendant la pandémie de coronavirus (COVID-19)? Display This Question:
- Options:
  - Très à l'aise (1)
  - Plutôt à l'aise (2)
  - Pas très à l'aise (3)
  - Vraiment pas à l'aise (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_comfort2`
- Label: If you decide to vote, how comfortable are you with the idea of voting
- Question: in person during the coronavirus (COVID-19) pandemic?
- Options:
  - Very comfortable (1)
  - Somewhat comfortable (2)
  - Not very comfortable (3)
  - Not comfortable at all (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_comfort2`
- Label: Si vous décidez de voter, à quel point êtes-vous à l'aise de voter en
- Question: personne pendant la pandémie de coronavirus (COVID-19)? Display This Question:
- Options:
  - Très à l'aise (1)
  - Plutôt à l'aise (2)
  - Pas très à l'aise (3)
  - Vraiment pas à l'aise (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_comfort3`
- Label: Regardless of how you voted, how comfortable were you with the idea
- Question: of voting in person during the coronavirus (COVID-19) pandemic?
- Options:
  - Very comfortable (1)
  - Somewhat comfortable (2)
  - Not very comfortable (3)
  - Not comfortable at all (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_comfort3`
- Label: Quelle que soit la façon dont vous avez voté, dans quelle mesure
- Question: étiez-vous à l'aise avec l'idée de voter en personne pendant la pandémie de coronavirus (COVID-19)? Vote preference in 2021 federal election Display This Question:
- Options:
  - Très à l'aise (1)
  - Plutôt à l'aise (2)
  - Pas très à l'aise (3)
  - Vraiment pas à l'aise (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_votechoice`
- Label: Which party do you think you will vote for?
- Question: Display This Choice: ________________________________________________
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_votechoice_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_votechoice`
- Label: Pour quel parti prévoyez-vous voter?
- Question: Display This Choice: response option in cps21_votechoice. Display This Question: which you... = Certain to vote
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_votechoice_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_votechoice_pr`
- Label: If you could vote in this election, which party do you think you
- Question: would vote for? Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_votechoice_pr_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_votechoice_pr`
- Label: Si vous pouviez voter lors de cette élection, pour quel parti
- Question: voteriez-vous? Display This Choice: each response option in cps21_votechoice_pr. Display This Question:
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_votechoice_pr_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_vote_unlikely`
- Label: If you decide to vote, which party do you think you will vote for?
- Question: Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_vote_unlikely_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_vote_unlikely`
- Label: Si vous décidez de voter, pour quel parti prévoyez-vous voter?
- Question: Display This Choice: response option in cps21_vote_unlikely. Display This Question: which you... = Unlikely to vote
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_vote_unlikely_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_vote_unlike_pr`
- Label: If you could vote in this election, and decided to vote, which
- Question: party do you think you would vote for? Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_vote_unlike_pr_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_vote_unlike_pr`
- Label: Si vous pouviez voter lors de cette élection et que vous
- Question: décidiez de le faire, pour quel parti voteriez-vous? Display This Choice: ________________________________________________ each response option in cps21_vote_unlike_pr.
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6) : cps21_vote_unlike_pr_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_v_advance`
- Label: For which party did you vote?
- Question: Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_v_advance_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_v_advance`
- Label: Pour quel parti avez-vous voté?
- Question: Display This Choice: response option in cps21_v_advance. Display This Question: And If
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_v_advance_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_vote_lean`
- Label: Is there a party you are leaning towards?
- Question: Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_vote_lean_6_TEXT
  - I do not intend to vote (7)
  - Don't know/ Prefer not to answer (8)

### `cps21_vote_lean`
- Label: Êtes-vous tenté(e) d'appuyer un parti en particulier?
- Question: Display This Choice: response option in cps21_vote_lean. Display This Question: which you... = Don't know/ Prefer not to answer
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_vote_lean_6_TEXT
  - Je ne prévois pas voter (7)
  - Je ne sais pas/Préfère ne pas répondre (8)

### `cps21_vote_lean_pr`
- Label: If you could vote in this election, is there a party you would be
- Question: leaning towards? Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_vote_lean_pr_6_TEXT
  - I do not intend to vote (7)
  - Don't know/ Prefer not to answer (8)

### `cps21_vote_lean_pr`
- Label: Si vous pouviez voter lors de l'élection, seriez-vous tenté(e)
- Question: d'appuyer un parti en particulier? Display This Choice: ________________________________________________ response option in cps21_vote_lean_pr.
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_vote_lean_pr_6_TEXT
  - Je ne prévois pas voter (7)
  - Je ne sais pas/Préfère ne pas répondre (8)

### `cps21_2nd_choice`
- Label: And which party would be your second choice?
- Question: Display This Choice: Display This Choice: Display This Choice: Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_2nd_choice_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_2nd_choice`
- Label: Et quel parti serait votre deuxième choix?
- Question: Display This Choice: Display This Choice: Display This Choice: Display This Choice:
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_2nd_choice_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_2nd_choice_pr`
- Label: And which party would be your second choice?
- Question: Display This Choice: Liberal Party Display This Choice: Conservative Party
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_2nd_choice_pr_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_2nd_choice_pr`
- Label: Et quel parti serait votre deuxième choix?
- Question: Display This Choice: Liberal Party Display This Choice: Conservative Party
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Autre parti (veuillez spécifier) (6): cps21_2nd_choice_pr_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_not_vote_for`
- Label: Are there any parties that you would absolutely not vote for?
- Question: (Select all that apply) Display This Choice: And If you could vote in this election, which party do you think you would vote for? != Liberal Party

### `cps21_not_vote_for_6_TEXT`
- Label: ▢ I could vote for any of the parties (cps21_not_vote_for_7 )
- Question: ▢ Don't know/ Prefer not to answer (cps21_not_vote_for_8)

### `cps21_not_vote_for`
- Label: Y a-t-il un ou des partis pour lesquels vous ne voteriez
- Question: absolument pas? (Veuillez sélectionner tous les partis qui s'appliquent) Display This Choice: And If you could vote in this election, which party do you think you would vote for? != Liberal Party

### `cps21_not_vote_for_6_TEXT`
- Label: ▢ Je ne prévois pas voter (cps21_not_vote_for_7)
- Question: ▢ Je ne sais pas/Préfère ne pas répondre (cps21_not_vote_for_8) response option in cps21_not_vote_for. Display This Question: And If

### `cps21_not_vote_for_w`
- Label: Why would you not vote for this party/these parties? (Select all
- Question: that apply) ▢ I don’t like the leader (cps21_not_vote_for_w_1) ▢ I don't like the policies of the party (cps21_not_vote_for_w_2) ▢ I don’t like the type of people who support the party

### `cps21_not_vote_for_w_4_TEXT`
- Label: ▢ Don't know/ Prefer not to answer (cps21_not_vote_for_w_5)

### `cps21_not_vote_for_w`
- Label: Pourquoi ne voteriez-vous pas pour ce(s) parti(s)? (Veuillez
- Question: sélectionner tous les réponses qui s'appliquent) ▢ Je n'aime pas le chef ou la cheffe (cps21_not_vote_for_w_1) ▢ Je n'aime pas les politiques du parti (cps21_not_vote_for_w_2) ▢ Je n'aime pas les personnes qui soutiennent le parti

### `cps21_not_vote_for_w_4_TEXT`
- Label: ▢ Je ne sais pas/ Préfère ne pas répondre (cps21_not_vote_for_w_5)
- Question: Ssatisfaction with federal government

### `cps21_fed_gov_sat`
- Label: How satisfied are you with the performance of the federal
- Question: government under Justin Trudeau?
- Options:
  - Very satisfied (1)
  - Fairly satisfied (2)
  - Not very satisfied (3)
  - Not at all satisfied (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_fed_gov_sat`
- Label: Quel est votre niveau de satisfaction quant à la performance du
- Question: gouvernement fédéral sous Justin Trudeau? Party rating
- Options:
  - Très satisfait (1)
  - Assez satisfait (2)
  - Pas très satisfait (3)
  - Pas du tout satisfait (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `cps21_party_rating`
- Label: How do you feel about the federal political parties below? Set the
- Question: slider to a number from 0 to 100, where 0 means you really dislike the party and 100 means you really like the party. Really dislike Really like Don't know/ Prefer not to

### `cps21_party_rating`
- Label: Que pensez-vous des partis fédéraux énumérés ci-dessous?
- Question: Veuillez glisser le curseur sur un chiffre entre 0 et 100, où 0 indique que vous n'aimez vraiment pas du tout ce parti et 100 que vous aimez vraiment beaucoup ce parti. Je n'aime J'aime Je ne sais vraiment pas vraiment pas/préfère ne

### `cps21_lead_rating`
- Label: How do you feel about the federal party leaders below? Set the
- Question: slider to a number from 0 to 100, where 0 means you really dislike the leader and 100 means you really like the leader. Really dislike Really like Don't know the leader

### `cps21_lead_rating`
- Label: Que pensez-vous des chefs de partis fédéraux énumérés ci-
- Question: dessous? Veuillez glisser le curseur sur un chiffre entre 0 et 100, où 0 indique que vous n'aimez vraiment pas du tout ce chef et 100 que vous aimez vraiment beaucoup ce chef. Je n'aime J'aime Je ne connais

### `cps21_cand_rating_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_cand_rating_t`
- Label: Timing
- Question: Local candidate ratings Display This Question:
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_cand_rating`
- Label: How do you feel about the candidates in your local riding? Set
- Question: the slider to a number from 0 to 100, where 0 means you really dislike the candidate and 100 means you really like the candidate. Really dislike Really like Don't know candidate/ Not

### `cps21_cand_rating`
- Label: Que pensez-vous des candidat(e)s dans votre circonscription
- Question: locale? Veuillez glisser le curseur sur un chiffre entre 0 et 100, où 0 indique que vous n'aimez vraiment pas du tout cette personne et 100 que vous aimez vraiment beaucoup cette personne. Je n'aime J'aime Je ne connais

### `cps21_lr_parties`
- Label: In politics, people sometimes talk of left and right. Where would you
- Question: place the federal political parties on a scale from 0 to 10 where 0 means the left and 10 means the right? Left Right Unsure 0 1 2 3 4 5 6 7 8 9 10

### `cps21_lr_parties`
- Label: En politique, on parle parfois de gauche et de droite. Où placeriez-
- Question: vous les partis politiques fédéraux sur une échelle de 0 à 10, où 0 indique la gauche et 10 la droite? Gauche Droite Pas certain(e) 0 1 2 3 4 5 6 7 8 9 10

### `cps21_lr_scale_bef_1`
- Label: Using the same scale where 0 means the left and 10 means the
- Question: right, where would you place yourself on this scale? Left Right Unsure 0 1 2 3 4 5 6 7 8 9 10

### `cps21_lr_scale_bef_1`
- Label: En politique, on parle parfois de gauche et de droite. Où vous
- Question: placeriez-vous sur cette échelle? Gauche Droite Pas certain(e) 0 1 2 3 4 5 6 7 8 9 10 Leader impressions

### `cps21_lead_int`
- Label: Which party leader(s) below do you think is/are intelligent? (Select all
- Question: that apply) ▢ Justin Trudeau (cps21_lead_int_1) ▢ Erin O'Toole (cps21_lead_int_2) ▢ Jagmeet Singh (cps21_lead_int_3)

### `cps21_lead_int`
- Label: Parmi les chefs de partis fédéraux énumérés ci-dessous, le(s)quel(s)
- Question: trouvez-vous intelligent(s)? (Sélectionnez tous ceux qui s'appliquent) ▢ Justin Trudeau (cps21_lead_int_1) ▢ Erin O'Toole (cps21_lead_int_2) ▢ Jagmeet Singh (cps21_lead_int_3)

### `cps21_lead_strong`
- Label: Which party leader(s) below do you think provide(s) strong
- Question: leadership? (Select all that apply) ▢ Justin Trudeau (cps21_lead_strong_1) ▢ Erin O'Toole (cps21_lead_strong_2) ▢ Jagmeet Singh (cps21_lead_strong_3)

### `cps21_lead_strong`
- Label: Parmi les chefs de partis fédéraux ci-dessous, le(s)quel(s)
- Question: manifeste(nt) un leadership fort ? (Sélectionnez tous ceux qui s'appliquent) ▢ Justin Trudeau (cps21_lead_strong_1) ▢ Erin O'Toole (cps21_lead_strong_2) ▢ Jagmeet Singh (cps21_lead_strong_3)

### `cps21_lead_trust`
- Label: Which party leader(s) below do you think is/are trustworthy? (Select
- Question: all that apply) ▢ Justin Trudeau (cps21_lead_trust_1) ▢ Erin O'Toole (cps21_lead_trust_2) ▢ Jagmeet Singh (cps21_lead_trust_3)

### `cps21_lead_trust`
- Label: Parmi les chefs de partis fédéraux énumérés ci-dessous, le(s)quel(s)
- Question: trouvez-vous digne(s) de confiance? (Sélectionnez tous ceux qui s'appliquent) ▢ Justin Trudeau (cps21_lead_trust_1) ▢ Erin O'Toole (cps21_lead_trust_2) ▢ Jagmeet Singh (cps21_lead_trust_3)

### `cps21_lead_cares`
- Label: Which party leader(s) below do you think really care(s) about
- Question: people like you? (Select all that apply) ▢ Justin Trudeau (cps21_lead_cares_1) ▢ Erin O'Toole (cps21_lead_cares_2) ▢ Jagmeet Singh (cps21_lead_cares_3)

### `cps21_lead_cares`
- Label: Parmi les chefs de partis fédéraux énumérés ci-dessous,
- Question: le(s)quel(s) se souci(ent) vraiment des gens comme vous? (Sélectionnez tous ceux qui s'appliquent) ▢ Justin Trudeau (cps21_lead_cares_1) ▢ Erin O'Toole (cps21_lead_cares_2)

### `cps21_spend_educ`
- Label: How much should the federal government spend on education?
- Options:
  - Spend less (1)
  - Spend about the same as now (2)
  - Spend more (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_spend_educ`
- Label: Combien le gouvernement fédéral devrait-il dépenser en
- Question: éducation?
- Options:
  - Dépenser moins (1)
  - Dépenser à peu près autant (2)
  - Dépenser plus (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_spend_env`
- Label: How much should the federal government spend on the
- Question: environment?
- Options:
  - Spend less (1)
  - Spend about the same as now (2)
  - Spend more (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_spend_env`
- Label: Combien le gouvernement fédéral devrait-il dépenser en
- Question: environnement?
- Options:
  - Dépenser moins (1)
  - Dépenser à peu près autant (2)
  - Dépenser plus (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_spend_just_law`
- Label: How much should the federal government spend on
- Question: ${e://Field/justice_law}?
- Options:
  - Spend less (1)
  - Spend about the same as now (2)
  - Spend more (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_spend_just_law`
- Label: Combien le gouvernement fédéral devrait-il dépenser
- Question: pour ${e://Field/justice_law_fr}?
- Options:
  - Dépenser moins (1)
  - Dépenser à peu près autant (2)
  - Dépenser plus (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_spend_defence`
- Label: How much should the federal government spend on defence?
- Options:
  - Spend less (1)
  - Spend about the same as now (2)
  - Spend more (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_spend_defence`
- Label: Combien le gouvernement fédéral devrait-il dépenser en
- Question: défense?
- Options:
  - Dépenser moins (1)
  - Dépenser à peu près autant (2)
  - Dépenser plus (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_spend_imm_min`
- Label: How much should the federal government spend
- Question: on immigrants and minorities?
- Options:
  - Spend less (1)
  - Spend about the same as now (2)
  - Spend more (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_spend_imm_min`
- Label: Combien le gouvernement fédéral devrait-il dépenser pour les
- Question: immigrants et les minorités?
- Options:
  - Dépenser moins (1)
  - Dépenser à peu près autant (2)
  - Dépenser plus (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_spend_rec_indi`
- Label: How much should the federal government spend on
- Question: reconciliation with Indigenous Peoples?
- Options:
  - Spend less (1)
  - Spend about the same as now (2)
  - Spend more (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_spend_rec_indi`
- Label: Combien le gouvernement fédéral devrait-il dépenser pour la
- Question: réconciliation avec les Peuples autochtones?
- Options:
  - Dépenser moins (1)
  - Dépenser à peu près autant (2)
  - Dépenser plus (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_spend_afford_h`
- Label: How much should the federal government spend on
- Question: affordable housing?
- Options:
  - Spend less (1)
  - Spend about the same as now (2)
  - Spend more (3)
  - Don't know/Prefer not to answer (4)

### `cps21_spend_afford_h`
- Label: Combien le gouvernement fédéral devrait-il dépenser pour
- Question: les logements abordables?
- Options:
  - Dépenser moins (1)
  - Dépenser à peu près autant (2)
  - Dépenser plus (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_spend_nation_c`
- Label: How much should the federal government spend on a
- Question: national childcare system?
- Options:
  - Spend less (1)
  - Spend about the same as now (2)
  - Spend more (3)
  - Don't know/Prefer not to answer (4)

### `cps21_spend_nation_c`
- Label: Combien le gouvernement fédéral devrait-il dépenser pour un
- Question: système national de garderies? block contained a corresponding government spending question(s): • Question block Government spending-education: cps21_spend_educ • Question block Government spending-environment: cps21_spend_env
- Options:
  - Dépenser moins (1)
  - Dépenser à peu près autant (2)
  - Dépenser plus (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_spend_imm_min`
- Label: • Question block Government spending-reconciliation with indigenous peoples:

### `cps21_spend_rec_indi`
- Label: • Question block Government spending-affordable housing/national childcare
- Question: system: cps21_spend_afford_h, cps21_spend_nation_c The following variables give the display order for each block, respectively: • FL_60_DO_Governmentspending_educ • FL_60_DO_Governmentspending_envi

### `cps21_pos_intro`
- Label: Now we will ask your opinion about a number of political issues. For
- Question: each one, please tell us whether you strongly disagree, somewhat disagree, neither agree nor disagree, somewhat agree or strongly agree.

### `cps21_pos_intro`
- Label: Nous allons maintenant vous demander votre avis sur certains
- Question: enjeux politiques. Pour chaque enjeu, veuillez nous indiquer si vous êtes fortement en désaccord, plutôt en désaccord, ni en accord, ni en désaccord, plutôt d'accord ou fortement d'accord.

### `cps21_pos_mailtrust`
- Label: Voting by mail is equally as trustworthy as voting in person.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_mailtrust`
- Label: Voter par la poste est tout aussi fiable que voter en personne.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_fptp`
- Label: Canada should change its electoral system from “First Past the Post”
- Question: to a “proportional representation” system.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_fptp`
- Label: Le Canada devrait changer son système électoral afin de passer du
- Question: mode de scrutin actuel à un système de "représentation proportionnelle".
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_life`
- Label: Individuals who are terminally ill should be allowed to end their lives
- Question: with the assistance of a doctor.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_life`
- Label: Les personnes en phase terminale devraient pouvoir mettre fin à leurs
- Question: jours avec l'aide d'un médecin.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_cannabis`
- Label: Possession of cannabis should be a criminal offence.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_cannabis`
- Label: La possession de cannabis devrait être une infraction pénale.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_carbon`
- Label: To help reduce greenhouse gas emissions, the federal
- Question: government should continue the carbon tax.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_carbon`
- Label: Afin d'aider à réduire les émissions de gaz à effet de serre, le
- Question: gouvernement fédéral devrait maintenir la taxe sur le carbone.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_energy`
- Label: The federal government should do more to help Canada’s energy
- Question: sector, including building oil pipelines.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_energy`
- Label: Le gouvernement fédéral devrait en faire davantage afin d'aider le
- Question: secteur énergétique canadien, notamment en construisant des oléoducs.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_envreg`
- Label: Environmental regulation should be stricter, even if it leads to
- Question: consumers having to pay higher prices.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_envreg`
- Label: La réglementation environnementale devrait être plus stricte,
- Question: même si elle oblige les consommateurs à payer des prix plus élevés.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_jobs`
- Label: When there is a conflict between protecting the environment and
- Question: creating jobs, jobs should come first.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_jobs`
- Label: Lorsqu'il existe un conflit entre la protection de l'environnement et la
- Question: création d'emplois, les emplois devraient avoir la priorité.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_subsid`
- Label: The federal government should end all corporate and economic
- Question: development subsidies.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_subsid`
- Label: Le gouvernement fédéral devrait mettre fin à toutes les
- Question: subventions au développement des entreprises et de l'économie.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_pos_trade`
- Label: There should be more free trade with other countries, even if it hurts
- Question: some industries in Canada.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_pos_trade`
- Label: Il devrait y avoir plus de libre-échange avec d'autres pays, même si
- Question: cela nuit à certaines industries au Canada.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_covid_liberty`
- Label: The public health recommendations aimed at slowing the spread
- Question: of the COVID-19 virus are threatening my liberty.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `cps21_covid_liberty`
- Label: Les recommandations de la santé publique visant à ralentir la
- Question: propagation de la COVID-19 menacent ma liberté. in the Survey Flow to recieve 6 (out of 10) of the following questions: cps21_pos_fptp, cps21_pos_life, cps21_pos_cannabis, cps21_pos_carbon, cps21_pos_energy, cps21_pos_envreg, cps21_pos_jobs, cps21_pos_subsid, cps21_pos_trade, or
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `cps21_econ_retro_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_econ_retro_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_econ_retro`
- Label: Over the past year, has Canada's economy...
- Options:
  - Got better (1)
  - Stayed about the same (2)
  - Got worse (3)
  - Don’t know (4)
  - Prefer not to answer (5)

### `cps21_econ_retro`
- Label: Depuis un an, l'économie canadienne s'est-elle:
- Options:
  - Améliorée (1)
  - Restée à peu près la même (2)
  - Détériorée (3)
  - Je ne sais pas (4)
  - Préfère ne pas répondre (5)

### `cps21_econ_fed_bet_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_econ_fed_bet_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_econ_fed_bette`
- Label: Have the policies of the federal government made Canada's
- Question: economy...
- Options:
  - Better (1)
  - Worse (2)
  - Not made much difference (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_econ_fed_bette`
- Label: Les politiques du gouvernement fédéral ont contribué à...
- Question: Handling of issues by parties Display This Question:
- Options:
  - Améliorer l'économie canadienne (1)
  - Détérioré l'économie canadienne (2)
  - N'ont pas changé grand chose à l'économie canadienne (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_issue_handle`
- Label: Which party would do the best job at handling each of the
- Question: following issues? Don't know/ Liberal Bloc Green
- Options:
  - (1) (4) (5)

### `cps21_issue_handle`
- Label: Quel parti aborderait le mieux chacun de ces enjeux?
- Question: Je ne sais Parti Parti Bloc NPD Part pas/Préfère libéral conservateur québécois
- Options:
  - (1) (2) (4)
  - répondre (6)

### `cps21_most_seats`
- Label: For each of the parties below, how likely is each party to win the
- Question: most seats in the House of Commons? No chance at Absolutely Don't know/ all of winning certain to win Prefer not to the most seats the most seats answer

### `cps21_most_seats`
- Label: Quel parti a le plus de chances de gagner le plus grand nombre
- Question: de sièges à la Chambre des communes? Aucune Absolument Je ne sais chance de certain de pas/Préfère ne gagner le plus gagner le plus pas répondre

### `cps21_win_local`
- Label: For each of the parties below, how likely is each party to win the seat
- Question: in your own local riding? No chance at Absolutely Don't know/ all of winning certain to win Not Applicable your riding your riding

### `cps21_win_local`
- Label: Quel parti a le plus de chances de gagner le siège de votre
- Question: circonscription? Aucune Absolument Je ne sais chance de certain de pas/Préfère ne gagner le gagner le pas répondre

### `cps21__candidateref`
- Label: Which candidate do you want to win the seat in your riding?
- Question: Display This Choice:
- Options:
  - Liberal candidate in your riding (1)
  - Conservative candidate in your riding (2)
  - NDP candidate in your riding (3)
  - Bloc Québécois candidate in your riding (4)
  - Green candidate in your riding (5)
  - Another candidate in your riding (please specify) (6):

### `cps21__candidateref_6_TEXT`
- Options:
  - Don't know/ Prefer not to answer (7)

### `cps21__candidateref`
- Label: Quel candidat ou candidate aimeriez-vous voir gagner votre
- Question: circonscription? Display This Choice: ________________________________________________
- Options:
  - Candidat(e) libéral(e) dans votre circonscription (1)
  - Candidat(e) conservateur(rice) dans votre circonscription (2)
  - Candidat(e) néodémocrat(e) dans votre circonscription (3)
  - Candidat(e) bloquiste dans votre circonscription (4)
  - Candidat(e) vert(e) dans votre circonscription (5)
  - Un(e) autre candidat(e) (veuillez spécifier): (6): cps21__candidateref_6_TEXT
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_candidate_imag`
- Label: Imagine you were the only voter in the election. Which
- Question: candidate would you want to win in your riding? Display This Choice:
- Options:
  - Liberal candidate in your riding (1)
  - Conservative candidate in your riding (2)
  - NDP candidate in your riding (3)
  - Bloc Québécois candidate in your riding (4)
  - Green candidate in your riding (5)
  - Another candidate in your riding (please specify) (6):

### `cps21_candidate_imag_6_TEXT`
- Options:
  - Don't know/ Prefer not to answer (7)

### `cps21_candidate_imag`
- Label: Imaginez que vous êtes le seule électeur ou la seule électrice
- Question: dans votre circonscription. Quel candidat ou candidate aimeriez-vous voir gagner votre circonscription? Display This Choice:
- Options:
  - Candidat(e) libéral(e) dans votre circonscription (1)
  - Candidat(e) conservateur(rice) dans votre circonscription (2)
  - Candidat(e) néodémocrat(e) dans votre circonscription (3)
  - Candidat(e) bloquiste dans votre circonscription (4)
  - Candidat(e) vert(e) dans votre circonscription (5)
  - Un(e) autre candidat(e) (veuillez spécifier): (6):

### `cps21_candidate_imag_6_TEXT`
- Label: End of Block: Candidate Outcome (Imagine)
- Question: Preferred election outcome Display This Question:
- Options:
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_outcome_most`
- Label: Which election outcome would you most prefer?
- Options:
  - Liberal majority (1)
  - Conservative majority (2)
  - NDP majority (3)
  - Liberal minority (4)
  - Conservative minority (5)
  - NDP minority (6)
  - Other government (7): cps21_outcome_most_7_TEXT
  - Don't know/ Prefer not to answer (8)

### `cps21_outcome_most`
- Label: Quel résultat électoral préféreriez-vous le plus?
- Question: each response option in cps21_outcome_most. Display This Question:
- Options:
  - Majorité libérale (1)
  - Majorité conservatrice (2)
  - Majorité NPD (3)
  - Minorité libérale (4)
  - Minorité conservatrice (5)
  - Minorité NPD (6)
  - Autre gouvernement (7): cps21_outcome_most_7_TEXT
  - Je ne sais pas/Préfère ne pas répondre (8)

### `cps21_outcome_least`
- Label: Which election outcome would you least prefer?
- Question: Display This Choice: Display This Choice: Display This Choice: Display This Choice:
- Options:
  - Liberal majority (1)
  - Conservative majority (2)
  - NDP majority (3)
  - Liberal minority (4)
  - Conservative minority (5)
  - NDP minority (6)
  - Other government (7): cps21_outcome_least_7_TEXT
  - Don't know/ Prefer not to answer (8)

### `cps21_outcome_least`
- Label: Quel résultat d'élection aimeriez-vous le moins?
- Question: Display This Choice: Display This Choice: Display This Choice: Display This Choice:
- Options:
  - Majorité libérale (1)
  - Majorité conservatrice (2)
  - Majorité NPD (3)
  - Minorité libérale (4)
  - Minorité conservatrice (5)
  - Minorité NPD (6)
  - Autre gouvernement (7): cps21_outcome_least_7_TEXT
  - Je ne sais pas/Préfère ne pas répondre (8)

### `cps21_minority_gov`
- Label: Do you think minority governments are:
- Options:
  - A good thing (1)
  - A bad thing (2)
  - I am not sure (3)
  - Prefer not to answer (4)

### `cps21_minority_gov`
- Label: Pensez-vous que les gouvernements minoritaires sont...
- Question: Attitude towards immigrants and refugees
- Options:
  - Une bonne chose (1)
  - Une mauvaise chose (2)
  - Je ne suis pas certain(e) (3)
  - Préfère ne pas répondre (4)

### `cps21_imm`
- Label: Do you think Canada should admit:
- Options:
  - More immigrants (1)
  - Fewer immigrants (2)
  - About the same number of immigrants as now (3)
  - Don’t know/ Prefer not to answer (4)

### `cps21_imm`
- Label: Pensez-vous que le Canada devrait admettre:
- Options:
  - Plus d'immigrants (1)
  - Moins d'immigrants (2)
  - À peu près le même nombre d'immigrants (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_refugees`
- Label: Do you think Canada should admit:
- Options:
  - More refugees (1)
  - Fewer refugees (2)
  - About the same number of refugees as now (3)
  - Don’t know/ Prefer not to answer (4)

### `cps21_refugees`
- Label: Pensez-vous que le Canada devrait admettre:
- Question: FL_67_DO_Immigration and FL_67_DO_Refugees, give the display order for each variable, respecitvley. Attention Check
- Options:
  - Plus de réfugiés (1)
  - Moins de réfugiés (2)
  - À peu près le même nombre de réfugiés (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_attcheck`
- Label: You probably have a favourite colour. But we are more interested in
- Question: making sure you’re doing the survey carefully, so please just select the colour brown here.
- Options:
  - Orange (1)
  - Blue (2)
  - Green (3)
  - Brown (4)
  - Yellow (5)
  - Black (6)
  - Purple (7)
  - White (8)
  - Red (9)
  - Don't know/ Prefer not to answer (10)

### `cps21_attcheck`
- Label: Vous avez probablement une couleur préférée, mais nous sommes
- Question: plus intéressés à nous assurer que vous répondez au sondage avec attention. Par conséquent, veuillez sélectionner la couleur brun ici. Efficacy
- Options:
  - Orange (1)
  - Bleu (2)
  - Vert (3)
  - Brun (4)
  - Jaune (5)
  - Noir (6)
  - Violet (7)
  - Blanc (8)
  - Rouge (9)
  - Je ne sais pas/Préfère ne pas répondre (10)

### `cps21_eff_intro`
- Label: Now we will ask your opinion about a few more political issues. For
- Question: each one, please tell us whether you strongly disagree, somewhat disagree, somewhat agree or strongly agree.

### `cps21_eff_intro`
- Label: Nous allons maintenant vous demander votre avis sur certains enjeux
- Question: politiques. Pour chaque énoncé, veuillez nous indiquer si vous êtes fortement en désaccord, plutôt en désaccord, plutôt d'accord ou fortement d'accord.

### `cps21_govt_confusing`
- Label: Sometimes, politics and government seem so complicated that
- Question: a person like me can't really understand what's going on.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Somewhat agree (3)
  - Strongly agree (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_govt_confusing`
- Label: Parfois, la politique et le gouvernement semblent si
- Question: compliqués qu'une personne comme moi ne peut pas vraiment comprendre ce qui se passe.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Plutôt d'accord (3)
  - Fortement d'accord (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_govt_say`
- Label: People like me don't have any say about what the government does.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Somewhat agree (3)
  - Strongly agree (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_govt_say`
- Label: Les gens comme moi n'ont rien à dire sur ce que le gouvernement
- Question: fait. Attitudes towards politicians
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Plutôt d'accord (3)
  - Fortement d'accord (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_pol_eth`
- Label: It is important that politicians behave ethically in office.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Somewhat agree (3)
  - Strongly agree (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_pol_eth`
- Label: Il est important que les politiciens se comportent de façon éthique dans
- Question: l'exercice de leurs fonctions. cps21_govt_say, and cps21_pol_eth. The variables FL_69_DO_Efficacy_understand, FL_69_DO_Efficacy_sayingovernmen, and FL_69_DO_Efficacy_politicianethi, give the display order for each variable,
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Plutôt d'accord (3)
  - Fortement d'accord (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_lib_promises`
- Label: Justin Trudeau kept the election promises he made in 2019.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Somewhat agree (3)
  - Strongly agree (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_lib_promises`
- Label: Justin Trudeau a tenu les promesses électorales qu'il avait faites
- Question: en 2019. News consumption
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Plutôt d'accord (3)
  - Tout à fait d'accord (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_news_cons`
- Label: On average, how much time do you usually spend watching,
- Question: reading, and listening to news each day?
- Options:
  - None (1)
  - 1-10 minutes (2)
  - 11-30 minutes (3)
  - 31-60 minutes (4)
  - Between 1 and 2 hours (5)
  - More than 2 hours (6)
  - Don't know/ Prefer not to answer (7)

### `cps21_news_cons`
- Label: En moyenne, combien de temps passez-vous chaque jour à lire,
- Question: regarder et écouter les nouvelles? Political knowledge
- Options:
  - 0 minutes (1)
  - 1-10 minutes (2)
  - 11-30 minutes (3)
  - 31-60 minutes (4)
  - 1 à 2 heures (5)
  - Plus de 2 heures (6)
  - Je ne sais pas/Préfère ne pas répondre (7)

### `cps21_premier_name_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_premier_name_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_premier_name`
- Label: We would like to see how widely known some political figures
- Question: are. Please answer off the top of your head without checking online. Do you happen to recall the name of the Premier of your Province? Display This Choice: Display This Choice:
- Options:
  - Brian Pallister (1)
  - Gregory Selinger (2)
  - François Legault (3)
  - Philippe Couillard (4)
  - Iain Rankin (5)
  - Tim Houston (6)
  - Blaine Higgs (7)
  - Brian Gallant (8)
  - John Horgan (9)
  - Christy Clark (10)
  - Dennis King (11)
  - Wade MacLauchlan (12)
  - Scott Moe (13)
  - Brad Wall (14)
  - Jason Kenney (15)
  - Rachel Notley (16)
  - Andrew Furey (17)
  - Dwight Ball (18)
  - Caroline Cochrane (19)
  - Bob McLeod (20)
  - Sandy Silver (21)
  - Darrell Pasloski (22)
  - Joe Savikataaq (23)
  - Paul Quassa (24)
  - Doug Ford (25)
  - Kathleen Wynne (26)
  - Doug Ford (27)
  - Brian Pallister (28)
  - François Legault (29)
  - Tim Houston (30)
  - Blaine Higgs (31)
  - John Horgan (32)
  - Dennis King (33)
  - Scott Moe (34)
  - Jason Kenney (35)
  - Andrew Furey (36)
  - Caroline Cochrane (37)
  - Sandy Silver (38)
  - Joe Savikataaq (39)
  - I don't know (40)
  - Prefer not to answer (41)

### `cps21_premier_name`
- Label: Nous aimerions savoir à quel point certaines personnalités
- Question: politiques sont connues. Veuillez écrire la première réponse qui vous vient en tête sans vérifier en ligne. Vous souvenez-vous du nom du premier ministre ou de la première ministre de votre province?
- Options:
  - Brian Pallister (1)
  - Gregory Selinger (2)
  - François Legault (3)
  - Philippe Couillard (4)
  - Iain Rankin (5)
  - Tim Houston (6)
  - Blaine Higgs (7)
  - Brian Gallant (8)
  - John Horgan (9)
  - Christy Clark (10)
  - Dennis King (11)
  - Wade MacLauchlan (12)
  - Scott Moe (13)
  - Brad Wall (14)
  - Jason Kenney (15)
  - Rachel Notley (16)
  - Andrew Furey (17)
  - Dwight Ball (18)
  - Caroline Cochrane (19)
  - Bob McLeod (20)
  - Sandy Silver (21)
  - Darrell Pasloski (22)
  - Joe Savikataaq (23)
  - Paul Quassa (24)
  - Doug Ford (25)
  - Kathleen Wynne (26)
  - Doug Ford (27)
  - Brian Pallister (28)
  - François Legault (29)
  - Tim Houston (30)
  - Blaine Higgs (31)
  - John Horgan (32)
  - Dennis King (33)
  - Scott Moe (34)
  - Jason Kenney (35)
  - Andrew Furey (36)
  - Caroline Cochrane (37)
  - Sandy Silver (38)
  - Joe Savikataaq (39)
  - Je ne sais pas (40)
  - Préfère ne pas répondre (41)

### `cps21_finmin_name_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_finmin_name_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_finmin_name`
- Label: What is the name of the federal Minister of Finance?
- Options:
  - Chrystia Freeland (1)
  - Carolyn Bennett (2)
  - Marc Garneau (3)
  - Bill Blair (4)
  - Mary Ng (5)
  - I don't know (6)
  - Prefer not to answer (7)

### `cps21_finmin_name`
- Label: Quel est le nom du/de la ministre fédéral(e) des finances?
- Question: each response option in cps21_finmin_name.
- Options:
  - Chrystia Freeland (1)
  - Carolyn Bennett (2)
  - Marc Garneau (3)
  - Bill Blair (4)
  - Mary Ng (5)
  - Je ne sais pas (6)
  - Préfère ne pas répondre (7)

### `cps21_govgen_name_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_govgen_name_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `cps21_govgen_name`
- Label: What is the name of the Governor-General of Canada?
- Options:
  - Julie Payette (1)
  - David Johnston (2)
  - Mary Simon (3)
  - Bill Morneau (4)
  - I don't know (5)
  - Prefer not to answer (6)

### `cps21_govgen_name`
- Label: Quel est le nom du/de la gouverneur(e) général(e) du Canada?
- Question: each response option cps21_govgen_name. Political participation
- Options:
  - Julie Payette (1)
  - David Johnston (2)
  - Mary Simon (3)
  - Bill Morneau (4)
  - Je ne sais pas (5)
  - Préfère ne pas répondre (6)

### `cps21_volunteer`
- Label: In the past 12 months, how many times did you volunteer for a group
- Question: or organization such as a school, a religious organization, or sports or community associations?
- Options:
  - Never (1)
  - Just once (2)
  - A few times (3)
  - More than five times (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_volunteer`
- Label: Au cours des 12 derniers mois, combien de fois avez-vous fait du
- Question: bénévolat pour un groupe ou un organisme comme une école, une organisation religieuse ou une association sportive ou communautaire? Duty to vote
- Options:
  - Jamais (1)
  - Juste une fois (2)
  - Quelques fois (3)
  - Plus de cinq fois (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_duty_choice`
- Label: People have different views about voting. For some, voting is a
- Question: duty. They feel that they should vote in every election. For others, voting is a choice. They only vote when they feel strongly about that election. For you personally, is voting first and foremost a Duty or a Choice?
- Options:
  - Duty (1)
  - Choice (2)
  - Don't know/ Prefer not to answer (3)

### `cps21_duty_choice`
- Label: Les gens ont des conceptions différentes du vote. Pour certains,
- Question: voter est un devoir. Ils croient qu'ils devraient voter à chaque élection. Pour d'autres, voter est un choix. Ils votent seulement lorsqu'une élection les préoccupe vraiment. Pour vous personnellement, est-ce que voter est un devoir ou un choix? Quebec sovereignty
- Options:
  - Devoir (1)
  - Choix (2)
  - Je ne sais pas/Préfère ne pas répondre (3)

### `cps21_quebec_sov`
- Label: Are you very favourable, somewhat favourable, somewhat
- Question: opposed, or very opposed to Quebec sovereignty, that is Quebec is no longer a part of Canada?
- Options:
  - Very favourable (1)
  - Somewhat favourable (2)
  - Somewhat opposed (3)
  - Very opposed (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_quebec_sov`
- Label: Êtes-vous très favorable, plutôt favorable, plutôt opposé(e) ou très
- Question: opposé(e) à la souveraineté du Québec, c'est-à-dire que le Québec ne fasse plus partie du Canada? Personal finances
- Options:
  - Très favorable (1)
  - Plutôt favorable (2)
  - Plutôt opposé(e) (3)
  - Très opposé(e) (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_own_fin_retro`
- Label: Over the past year, has your financial situation:
- Options:
  - Got better (1)
  - Stayed about the same (2)
  - Got worse (3)
  - Don’t know/ Prefer not to answer (4)

### `cps21_own_fin_retro`
- Label: Pendant la dernière année, votre situation financière s'est-elle :
- Options:
  - Améliorée (1)
  - Restée à peu près la même (2)
  - Détériorée (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_ownfinanc_fed`
- Label: Have the policies of the federal government made your
- Question: financial situation...
- Options:
  - Better (1)
  - Worse (2)
  - Not made much difference (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_ownfinanc_fed`
- Label: Les politiques du gouvernement fédéral ont contribué à...
- Options:
  - Améliorer votre situation financière (1)
  - Détériorer votre situation financière (2)
  - Pas changé grand chose à votre situation financière (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_own_fin_future`
- Label: Over the next year, do you think your financial situation will:
- Options:
  - Get better (1)
  - Stay about the same (2)
  - Get worse (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_own_fin_future`
- Label: Lors de la prochaine année, pensez-vous que votre situation
- Question: financière va : COVID relief programs
- Options:
  - S’améliorer (1)
  - Rester à peu près la même (2)
  - Se détériorer (3)
  - Ne sais pas/Préfère ne pas répondre (4)

### `cps21_covidrelief`
- Label: Have you applied for any of the following COVID relief programs?
- Question: Please select all that apply. ▢ Canada Emergency Response Benefit (CERB) (cps21_covidrelief__1) ▢ Canada Emergency Student Benefit (CESB) (cps21_covidrelief__2) ▢ Canada Recovery Benefit (CRB) (cps21_covidrelief__3)

### `cps21_covidrelief`
- Label: Avez-vous demandé l'une de ces prestations d'urgence en lien
- Question: avec la COVID-19? Sélectionnez toutes celles qui s'appliquent. ▢ Prestation canadienne d’urgence (PCU) (cps21_covidrelief__1) ▢ Prestation canadienne d'urgence pour les étudiants (PCUE) (cps21_covidrelief__2)

### `cps21_groupdiscrim`
- Label: How much discrimination is there in Canada against each of the
- Question: following groups? A A Don't know/ None great A lot moderate A little Prefer not
- Options:
  - (1) (3) (6)
  - o o o o o

### `cps21_groupdiscrim`
- Label: À quel point les groupes suivants font-ils face à de la
- Question: discrimination au Canada? Je ne sais Pas
- Options:
  - e (6)

### `cps21_prov_gov_sat`
- Label: How satisfied are you with the performance of your provincial
- Question: government under ${e://Field/premier}?
- Options:
  - Very satisfied (1)
  - Fairly satisfied (2)
  - Not very satisfied (3)
  - Not at all satisfied (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_prov_gov_sat`
- Label: Quel est votre niveau de satisfaction de la performance du
- Question: gouvernement provincial sous ${e://Field/premier}? Pandemic-related atttidue
- Options:
  - Très satisfait(e) (1)
  - Assez satisfait(e) (2)
  - Pas très satisfait(e) (3)
  - Pas du tout satisfait(e) (4)
  - Je ne sais pas/Préfère ne pas répondre (5)

### `cps21_covid_sat`
- Label: How satisfied are you with how each of the following have handled
- Question: the coronavirus outbreak? Don't Very Fairly Not very Not at all know/ satisfied satisfied satisfied satisfied Prefer not

### `cps21_covid_sat`
- Label: À quel point êtes-vous satisfait(e) de la façon dont les
- Question: gouvernements suivants ont géré la pandémie de coronavirus? Je ne sais Pas du Très Assez Pas très pas/Préfère

### `cps21_vaccine_mandat`
- Label: Should vaccination be required to:
- Question: Don't Strongly Somewhat Strongly Somewhat know/Pref disagree disagree agree
- Options:
  - (1) (2) (4)
  - answer (5)
  - o o o o

### `cps21_vaccine_mandat`
- Label: La vaccination devrait être requise pour…
- Question: Fortement Je ne sais Plutôt en Plutôt Fortement en pas/Préfère désaccord d'accord d'accord
- Options:
  - (2) (3) (4)
  - (1) répondre (5)

### `cps21_vaccine1`
- Label: Have you been vaccinated?
- Options:
  - Yes, with two or more doses (1)
  - Yes, with one dose so far (2)
  - No (3)
  - Prefer not to answer (4)

### `cps21_vaccine1`
- Label: Êtes-vous vacciné(e) contre la COVID-19?
- Question: Display This Question:
- Options:
  - Oui, j’ai reçu deux doses ou plus du vaccin (1)
  - Oui, j’ai reçu une dose du vaccin jusqu’à présent (2)
  - Non (3)
  - Préfère ne pas répondre (4)

### `cps21_vaccine2`
- Label: Are you eligible to be vaccinated?
- Options:
  - Yes (1)
  - No (2)
  - Prefer not to answer (3)

### `cps21_vaccine2`
- Label: Êtes-vous éligible à la vaccination?
- Question: Display This Question: And Are you eligible to be vaccinated? != No
- Options:
  - Oui (1)
  - Non (2)
  - Préfère ne pas répondre (3)

### `cps21_vaccine3`
- Label: Are you:
- Question: Display This Choice:
- Options:
  - Waiting for a scheduled appointment (1)
  - Planning to get vaccinated (2)
  - Not planning to get vaccinated (3)
  - Unable to get vaccinated for health reasons (4)
  - Unsure what you will do (5)
  - Prefer not to answer (6)

### `cps21_vaccine3`
- Label: Êtes-vous:
- Question: Display This Choice: Partisanship
- Options:
  - En attente d’un rendez-vous prévu (1)
  - Prévoyez de vous faire vacciner (2)
  - Ne prévoyez pas de vous faire vacciner (3)
  - Dans l’incapacité de recevoir le vaccin pour des raisons de santé (4)
  - Incertain de ce que vous ferez (5)
  - Préfère ne pas répondre (6)

### `cps21_fed_id`
- Label: In federal politics, do you usually think of yourself as a:
- Question: Display This Choice:
- Options:
  - Liberal (1)
  - Conservative (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green (5)
  - Another party (please specify) (6): cps21_fed_id_6_TEXT
  - None of these (7)
  - Don't know/ Prefer not to answer (8)

### `cps21_fed_id`
- Label: En politique fédérale, vous considérez-vous habituellement comme
- Question: étant: Display This Choice: response option in cps21_fed_id. Display This Question:
- Options:
  - Libéral (1)
  - Conservateur (2)
  - NPD (3)
  - Bloquiste (4)
  - Vert(e) (5)
  - Un autre parti (Veuillez spécifier) (6): cps21_fed_id_6_TEXT
  - Aucun de ces partis (7)
  - Je ne sais pas/Préfère ne pas répondre (8)

### `cps21_fed_id_str`
- Label: How strongly ${e://Field/pid_en}
- Question: ${cps21_fed_id/ChoiceTextEntryValue/7} do you feel?
- Options:
  - Very strongly (1)
  - Fairly strongly (2)
  - Not very strongly (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_fed_id_str`
- Label: À quel point vous sentez-vous proche du
- Question: ${e://Field/pid_party_fr}${cps21_fed_id/ChoiceTextEntryValue/7}? Display This Question: And In which province or territory are you currently living? != Nunavut And If
- Options:
  - Très fortement (1)
  - Fortement (2)
  - Pas très fortement (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_prov_id`
- Label: In provincial politics, do you usually think of yourself as a:
- Question: Display This Choice: Display This Choice: Display This Choice: Or In which province or territory are you currently living? = Ontario
- Options:
  - Liberal (1)
  - NDP (2)
  - Green (3)
  - Coalition Avenir Québec (4)
  - Parti Québécois (5)
  - Québec Solidaire (6)
  - United Conservative (7)
  - Alberta Party (8)
  - Buffalo Party (9)
  - Saskatchewan Party (10)
  - Progressive Conservative (11)
  - People's Alliance (12)
  - Yukon Party (13)
  - Another party (please specify) (14): cps21_prov_id_14_TEXT
  - None of these (15)
  - Don't know/ Prefer not to answer (16)

### `cps21_prov_id`
- Label: En politique provinciale, vous considérez-vous habituellement comme
- Question: étant: Display This Choice: Display This Choice: Display This Choice:
- Options:
  - Parti libéral (1)
  - NPD (2)
  - Parti vert (3)
  - Coalition avenir Québec (4)
  - Parti québécois (5)
  - Québec solidaire (6)
  - United Conservative (7)
  - Alberta Party (8)
  - Buffalo Party (9)
  - Saskatchewan Party (10)
  - Parti progressiste-conservateur du Canada (11)
  - Alliance des gens du Nouveau-Brunswick (12)
  - Yukon Party (13)
  - Autre parti (veuillez spécifier) (14): cps21_prov_id_14_TEXT
  - Aucun de ces partis (15)
  - Je ne sais pas/Préfère ne pas répondre (16)

### `cps21_prov_id_str`
- Label: How
- Question: strongly ${cps21_prov_id/ChoiceGroup/SelectedChoicesTextEntry} do you feel?
- Options:
  - Very strongly (1)
  - Fairly strongly (2)
  - Not very strongly (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_prov_id_str`
- Label: À quel point vous sentez-vous proche du
- Question: ${cps21_prov_id/ChoiceGroup/SelectedChoicesTextEntry} ? Groups thermometers
- Options:
  - Très proche (1)
  - Assez proche (2)
  - Pas très proche (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `cps21_groups_therm`
- Label: How do you feel about the following groups? Set the slider to
- Question: any number from 0 to 100, where 0 means you really dislike the group and 100 means you really like the group. Really dislike Really like Don't know/ Prefer not to

### `cps21_groups_therm`
- Label: Que pensez-vous des différents groupes ci-dessous? Veuillez
- Question: glisser la barre sur un chiffre entre 0 et 100, où 0 indique que vous n'aimez vraiment pas du tout un groupe et 100 que vous aimez vraiment beaucoup un groupe. Je n'aime J'aime Je ne sais vraiment pas vraiment pas/préfère ne

### `cps21_spoil`
- Label: Have you ever intentionally spoiled your ballot in an election (e.g.
- Question: intentionally filled out your ballot so your vote would not be counted for any candidate)?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `cps21_spoil`
- Label: Avez-vous déjà intentionnellement annulé votre vote lors d'une élection
- Question: (c'est-à-dire que vous avez volontairement rempli votre bulletin de vote afin qu'il ne soit compté pour aucun candidat)? Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas / Préfère ne pas répondre (3)

### `cps21_turnout_2019`
- Label: Did you happen to vote in the last Federal election in 2019?
- Options:
  - Yes (1)
  - No (2)
  - Not eligible to vote in last election (3)
  - Don't know/ Prefer not to answer (4)

### `cps21_turnout_2019`
- Label: Avez-vous voté lors de la dernière élection fédérale en 2019?
- Question: Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Pas éligible au vote lors de la dernière élection (3)
  - Je ne sais pas / Préfère ne pas répondre (4)

### `cps21_vote_2019`
- Label: Which party did you vote for in the Federal election in 2019?
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - Another party (please specify) (6): cps21_vote_2019_6_TEXT
  - Don't know/ Prefer not to answer (7)

### `cps21_vote_2019`
- Label: Pour quel parti avez-vous voté lors de l'élection fédérale de 2019?
- Question: response option in cps21_vote_2019. Federal leaders’ debates Display This Question:
- Options:
  - Parti libéral (1)
  - Parti conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Un autre parti (Veuillez spécifier) (6): cps21_vote_2019_6_TEXT
  - Je ne sais pas / Préfère ne pas répondre (7)

### `cps21_debate_fr`
- Label: Did you happen to watch or listen to the French-language federal
- Question: leaders debate on Thursday September 2nd?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `cps21_debate_fr`
- Label: Avez-vous écouté ou regardé le débat des chefs fédéraux en français
- Question: le jeudi 2 septembre? Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/Préfère ne pas répondre (3)

### `cps21_debate_fr2`
- Label: And what about the French-language federal leaders' debate on
- Question: Wednesday, September 8th? Did you happen to watch or listen to it?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `cps21_debate_fr2`
- Label: Et le débat des chefs fédéraux en français le mercredi 8 septembre? L'avez-
- Question: vous regardé ou écouté? Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/Préfère ne pas répondre (3)

### `cps21_debate_en`
- Label: Did you happen to watch or listen to the English-language federal
- Question: leaders debate on Wednesday September 9th?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `cps21_debate_en`
- Label: Avez-vous écouté ou regardé le débat des chefs fédéraux en
- Question: anglais le lundi 7 octobre? Talk about politics
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/Préfère ne pas répondre (3)

### `cps21_talkpolitics`
- Label: Over the past week, did you talk about politics or public affairs with
- Question: any of the following people? (Select all that apply) ▢ People whose political views are different from yours (cps21_talkpolitics_1) ▢ People who support the Liberal Party (cps21_talkpolitics_2)

### `cps21_talkpolitics`
- Label: Lors de la dernière semaine, avez-vous parlé de politique ou
- Question: d’affaires publiques avec des personnes en provenance de ces groupes (Sélectionnez toutes les réponses qui s’appliquent) ▢ Des personnes avec des idées politiques différentes des vôtres (cps21_talkpolitics_1)

### `cps21_resident_pr2`
- Label: Below are some actions that some believe should be taken to
- Question: respond to those impacted by residential schools. Do you support or oppose the following actions?

### `cps21_resident_pr2`
- Label: Vous trouverez ci-dessous des mesures qui, selon certains,
- Question: devraient être prises pour répondre aux besoins des personnes touchées par les pensionnats. Êtes-vous en faveur ou contre les actions suivantes?

### `cps21_residential_2a`
- Label: Accelerate progress on the calls to action from the truth and
- Question: reconciliation commission.
- Options:
  - Strongly oppose (1)
  - Somewhat oppose (2)
  - Somewhat support (3)
  - Strongly support (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_residential_2a`
- Label: Accélérer la mise en œuvre des appels à l'action lancés par la
- Question: Commission de vérité et de réconciliation.
- Options:
  - Fortement contre (1)
  - Plutôt contre (2)
  - Plutôt en faveur (3)
  - Fortement en faveur (4)
  - Ne sais pas/Préfère ne pas répondre (5)

### `cps21_residential_2b`
- Label: Federal government funding to identify unmarked graves at all
- Question: former residential schools.
- Options:
  - Strongly oppose (1)
  - Somewhat oppose (2)
  - Somewhat support (3)
  - Strongly support (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_residential_2b`
- Label: Que le gouvernement fédéral finance l'identification des tombes
- Question: non marquées dans tous les anciens pensionnats.
- Options:
  - Fortement contre (1)
  - Plutôt contre (2)
  - Plutôt en faveur (3)
  - Fortement en faveur (4)
  - Ne sais pas / Préfère ne pas répondre (5)

### `cps21_residential_2c`
- Label: All governments ceasing court actions against residential school
- Question: survivors and first nations children.
- Options:
  - Strongly oppose (1)
  - Somewhat oppose (2)
  - Somewhat support (3)
  - Strongly support (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_residential_2c`
- Label: Que tous les gouvernements cessent les actions en justice
- Question: contre les survivants des pensionnats et les enfants des Premières nations.
- Options:
  - Fortement contre (1)
  - Plutôt contre (2)
  - Plutôt en faveur (3)
  - Fortement en faveur (4)
  - Ne sais pas/Préfère ne pas répondre (5)

### `cps21_residential_2d`
- Label: Renaming buildings and institutions that are named for people
- Question: who built or ran parts of the residential school system.
- Options:
  - Strongly oppose (1)
  - Somewhat oppose (2)
  - Somewhat support (3)
  - Strongly support (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_residential_2d`
- Label: Renommer les bâtiments et les institutions qui portent le nom de
- Question: personnes qui ont construit ou dirigé en partie le système de pensionnats. Religion
- Options:
  - Fortement contre (1)
  - Plutôt contre (2)
  - Plutôt en faveur (3)
  - Fortement en faveur (4)
  - Ne sais pas/Préfère ne pas répondre (5)

### `cps21_religion`
- Label: Please indicate your religion, if you have one?
- Options:
  - None/ Don't have one/ Atheist (1)
  - Agnostic (2)
  - Buddhist/ Buddhism (3)
  - Hindu (4)
  - Jewish/ Judaism/ Jewish Orthodox (5)
  - Muslim/ Islam (6)
  - Sikh/ Sikhism (7)
  - Anglican/ Church of England (8)
  - Baptist (9)
  - Catholic/ Roman Catholic/ RC (10)
  - Greek Orthodox/ Ukrainian Orthodox/ Russian Orthodox/ Eastern Orthodox (11)
  - Jehovah's Witness (12)
  - Lutheran (13)
  - Mormon/ Church of Jesus Christ of the Latter Day Saints (14)
  - Pentecostal/ Fundamentalist/ Born Again/ Evangelical (15)
  - Presbyterian (16)
  - Protestant (17)
  - United Church of Canada (18)
  - Christian Reformed (19)
  - Salvation Army (20)
  - Mennonite (21)
  - Other (please specify) (22): cps21_religion_22_TEXT
  - Don't know/ Prefer not to answer (23)

### `cps21_religion`
- Label: Veuillez indiquer, s'il vous plaît, votre religion, si vous en avez une?
- Question: (11) Display This Question: Orthodox Or Please indicate your religion, if you have one? = Muslim/ Islam
- Options:
  - Aucune/ N'en a pas une / Athée (1)
  - Agnostique (2)
  - Bouddhiste / Bouddhisme (3)
  - Hindou (4)
  - Juif / Judaïsme / Orthodoxe Juif (5)
  - Musulman (6)
  - Sikh / Sikhisme (7)
  - Anglicane / Eglise d'Angleterre (8)
  - Baptiste (9)
  - Catholique / Catholique Romaine / RC (10)
  - Orthodoxe Grec / Orthodoxe Ukrainien / Orthodoxe Russe / Orthodoxe de l'Est
  - Témoin de Jehovah (12)
  - Luthérien (13)
  - Mormon / Église de Jésus-Christ des Saints des Derniers Jours (14)
  - Pentecôtiste / Fondamentaliste / Né de nouveau / Évangélique (15)
  - Presbytérien (16)
  - Protestant (17)
  - Église Unie du Canada (18)
  - Réforme chrétienne (19)
  - Armée du Salut (20)
  - Mennonite (21)
  - Autre (Veuillez spécifier) (22): cps21_religion_22_TEXT
  - Je ne sais pas / Préfère ne pas répondre (23)

### `cps21_denomination`
- Label: Which specific ${e://Field/religion_EN} denomination are you a
- Question: member of? ________________________________________________________________

### `cps21_denomination`
- Label: À quelle dénomination spécifique ${e://Field/religion_fr} adhérez-
- Question: vous ? ________________________________________________________________ Display This Question:

### `cps21_rel_imp`
- Label: In your life, you would say religion is:
- Options:
  - Very important (1)
  - Somewhat important (2)
  - Not very important (3)
  - Not important at all (4)
  - Don't know/ Prefer not to answer (5)

### `cps21_rel_imp`
- Label: Dans votre vie, diriez-vous que la religion est:
- Question: Additional demographics
- Options:
  - Très importante (1)
  - Assez importante (2)
  - Pas très importante (3)
  - Pas importante du tout (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `cps21_bornin_canada`
- Label: Were you born in Canada?
- Options:
  - Yes (1)
  - No (2)
  - Don’t know/ Prefer not to say (3)

### `cps21_bornin_canada`
- Label: Êtes-vous né(e) au Canada?
- Question: Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas / Préfère ne pas répondre (3)

### `cps21_bornin_other`
- Label: What country were you born in?
- Options:
  - ▼ AFGHANISTAN (1) ... Don't know/ Prefer not to say (249)

### `cps21_bornin_other`
- Label: Dans quel pays êtes-vous né(e)?
- Question: Display This Question:
- Options:
  - ▼ AFGHANISTAN (1) ... Je ne sais pas / Préfère ne pas répondre (249)

### `cps21_imm_year`
- Label: In what year did you come to live in Canada?
- Options:
  - ▼ 1920 (1) ... Don't know/ Prefer not to answer (103)

### `cps21_imm_year`
- Label: En quelle année êtes-vous venu(e) vivre au Canada?
- Question: Display This Question:
- Options:
  - ▼ 1920 (1) ... Je ne sais pas / Préfère ne pas répondre (103)

### `cps21_immig_status`
- Label: Under which immigration category did you enter Canada or
- Question: become a permanent resident in Canada?
- Options:
  - Skilled worker or professional – principal applicant (1)
  - Skilled worker or professional – dependent (2)
  - Family class (3)
  - Provincial nominee – principal applicant (4)
  - Provincial nominee – dependent (5)
  - Refugee or protected person (6)
  - Business class – principal applicant (7)
  - Business class – dependent (8)
  - Canadian experience class – principal applicant (9)
  - Canadian experience class – dependent (10)
  - Caregiver (11)
  - Other (please specify) (12): cps21_immig_status_12_TEXT
  - Don't know (13)
  - Prefer not to answer (14)

### `cps21_immig_status`
- Label: Sous quelle catégorie d'immigration êtes-vous entré(e) au
- Question: Canada ou êtes-vous devenu(e) résident(e) permanent(e) au Canada ?
- Options:
  - Travailleur qualifié ou professionnel - demandeur principal (1)
  - Travailleur qualifié ou professionnel - personne à charge (2)
  - Parrainage familiale (3)
  - Candidat provincial - candidat principal (4)
  - Candidat provincial - personne à charge (5)
  - Réfugié ou personne protégée (6)
  - Classe affaires - demandeur principal (7)
  - Classe affaires - personne à charge (8)
  - Classe d'expérience canadienne - demandeur principal (9)
  - Classe d'expérience canadienne - personne à charge (10)
  - Soignant (11)
  - Autre (veuillez spécifier) (12): cps21_immig_status_12_TEXT
  - Ne sais pas (13)
  - Préfère ne pas répondre (14)

### `cps21_origin`
- Label: What are the ethnic or cultural origins of your ancestors? Please indicate
- Question: up to 5. Ancestors may have Indigenous origins, or origins that refer to different countries, or other origins that may not refer to different countries. For examples, refer to this list of ethnic or cultural origins.
- Options:
  - 1 (cps21_origin_1) ________________________________________________
  - 2 (cps21_origin_2) ________________________________________________
  - 3 (cps21_origin_3) ________________________________________________
  - 4 (cps21_origin_4) ________________________________________________
  - 5 (cps21_origin_5) ________________________________________________

### `cps21_origin`
- Label: Quelles sont les origines ethniques ou culturelles de vos ancêtres?
- Question: Veuillez en indiquer jusqu’à 5. Les ancêtres peuvent avoir des origines autochtones, des origines qui réfèrent à différents pays, ou d’autres origines qui peuvent ne pas référer à un pays. Pour des exemples, veuillez consulter cette liste d’origines ethniques ou culturelles.
- Options:
  - 1 (cps21_origin_1) ________________________________________________
  - 2 (cps21_origin_2) ________________________________________________
  - 3 (cps21_origin_3) ________________________________________________
  - 4 (cps21_origin_4) ________________________________________________
  - 5 (cps21_origin_5) ________________________________________________

### `cps21_vismin`
- Label: Do you identify as any of the following? (Please select all that apply)
- Question: ▢ Arab (cps21_vismin_1) ▢ Asian (cps21_vismin_2) ▢ Black (cps21_vismin_3) ▢ Indigenous (e.g. First Nations, Métis, Inuit, etc.) (cps21_vismin_4)

### `cps21_vismin`
- Label: Vous identifiez-vous à un ou plusieurs de ces groupes... (Veuillez
- Question: sélectionner toutes les réponses applicables): ▢ Arabe (cps21_vismin_1) ▢ Asiatique (cps21_vismin_2) ▢ Noir(e) (cps21_vismin_3)

### `cps21_two_spirit`
- Label: Are you Two-Spirit?
- Options:
  - Yes (1)
  - No (2)
  - I don't know/ Prefer not to answer (3)

### `cps21_two_spirit`
- Label: Êtes-vous bispirituel(le)?
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas / Préfère ne pas répondre (3)

### `cps21_sexuality`
- Label: Which of the following best represents how you think of yourself?
- Options:
  - Straight or heterosexual (1)
  - Gay or lesbian (2)
  - Bisexual (3)
  - Queer (4)
  - Something else (5): cps21_sexuality_5_TEXT
  - I am not sure yet (6)
  - I don't know what this question means (7)
  - Prefer not to answer (8)

### `cps21_sexuality`
- Label: Vous considérez-vous comme étant :
- Options:
  - Hétérosexuel(le) (1)
  - Gai ou lesbienne (2)
  - Bisexuel(le) (3)
  - Queer (4)
  - Quelque chose d'autres (5): cps21_sexuality_5_TEXT
  - Je ne sais pas encore (6)
  - Je ne comprend pas la question (7)
  - Préfère ne pas répondre (8)

### `cps21_language`
- Label: Which language(s) did you learn as a child and still understand
- Question: today? (Select all that apply) ▢ English (cps21_language_1) ▢ French (cps21_language_2) ▢ Indigenous language (please specify) (cps21_language_3):

### `cps21_language_3_TEXT`
- Label: ▢ Arabic (cps21_language_4)
- Question: ▢ Chinese, Cantonese, Mandarin (cps21_language_5) ▢ Filipino / Tagalog (cps21_language_6) ▢ German (cps21_language_7) ▢ Indian, Hindi, Gujarati (cps21_language_8)

### `cps21_language_17_TEXT`
- Label: ▢ Don't know/ Prefer not to answer (cps21_language_18)

### `cps21_language`
- Label: Quelle est la/les première(s) langue(s) que vous avez apprise(s) et
- Question: que vous comprenez encore? (Sélectionnez toutes celles qui s' appliquent) ▢ Anglais (cps21_language_1) ▢ Français (cps21_language_2) ▢ Langue autochtone (veuillez préciser) (cps21_language_3):

### `cps21_language_3_TEXT`
- Label: ▢ Arabe (cps21_language_4)
- Question: ▢ Chinois, cantonais, mandarin (cps21_language_5) ▢ Philippin / tagalog (cps21_language_6) ▢ Allemand (cps21_language_7) ▢ Indien, Hindi, Gujarati (cps21_language_8)

### `cps21_language_17_TEXT`
- Label: ▢ Je ne sais pas / Préfère ne pas répondre (cps21_language_18)

### `cps21_employment`
- Label: What is your employment status? Are you currently…
- Options:
  - Working for pay full-time (1)
  - Working for pay part-time (2)
  - Self employed (with or without employees) (3)
  - Retired (4)
  - Unemployed/ looking for work (5)
  - Student (6)
  - Caring for a family (7)
  - Disabled (8)
  - Student and working for pay (9)
  - Caring for family and working for pay (10)
  - Retired and working for pay (11)
  - Other (please specify) (12): cps21_employment_12_TEXT
  - Don't know/ Prefer not to answer (13)

### `cps21_employment`
- Label: Quel est votre statut d'emploi actuel?
- Options:
  - Salarié à temps plein (1)
  - Salarié à temps partiel (2)
  - Travailleur autonome (avec ou sans employés) (3)
  - À la retraite (4)
  - Au chômage/à la recherche d'un travail (5)
  - Étudiant (6)
  - En charge d'une famille (7)
  - Handicapé (8)
  - Étudiant et salarié (9)
  - En charge d'une famille et salarié (10)
  - À la retraite et salarié (11)
  - Autre (veuillez spécifier) (12): cps21_employment_12_TEXT
  - Je ne sais pas / Préfère ne pas répondre (13)

### `cps21_union`
- Label: Do you belong to a union?
- Options:
  - Yes (1)
  - No (2)
  - Don’t know/ Prefer not to say (3)

### `cps21_union`
- Label: Appartenez-vous à un syndicat?
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas / Préfère ne pas répondre (3)

### `cps21_children`
- Label: How many children, if any, do you have?
- Options:
  - ▼ 0 (1) ... Don't know/ Prefer not to answer (7)

### `cps21_children`
- Label: Combien d'enfants avez-vous?
- Question: Display This Question: Or How many children, if any, do you have? = 2 Or How many children, if any, do you have? = 3 Or How many children, if any, do you have? = 4
- Options:
  - ▼ 0 (1) ... Je ne sais pas / Préfère ne pas répondre (7)

### `cps21_children_atten`
- Label: Do any of your children normally attend any of the following?
- Question: (Select all that apply) ▢ Daycare (cps21_children_atten_1) ▢ Elementary school (cps21_children_atten_2) ▢ Middle or high school (cps21_children_atten_3)

### `cps21_children_atten`
- Label: L'un de vos enfants fréquente-t-il en temps normal l'un des
- Question: établissements suivants ? (Sélectionnez toutes les réponses qui s'appliquent) ▢ Garderie (cps21_children_atten_1) ▢ École primaire (cps21_children_atten_2) ▢ École secondaire (cps21_children_atten_3)

### `cps21_income_number`
- Label: What was your total household income, before taxes, for the
- Question: year 2020? Be sure to include income from all sources, to the nearest thousand dollars. For example, if your household had a total before-tax income of $71,336 in 2020, you would enter 71000. ________________________________________________________________

### `cps21_income_number`
- Label: Quel est le revenu total de votre ménage avant impôts en
- Question: 2020? Cela doit inclure toutes les sources de revenus au millier de dollars près. Par exemple, si votre revenus total avant impôts était de 71 336$ en 2020, entrez 71000. ________________________________________________________________

### `cps21_income_cat`
- Label: We don't need the exact amount; does your household income fall
- Question: into one of these broad categories?
- Options:
  - No income (1)
  - $1 to $30,000 (2)
  - $30,001 to $60,000 (3)
  - $60,001 to $90,000 (4)
  - $90,001 to $110,000 (5)
  - $110,001 to $150,000 (6)
  - $150,001 to $200,000 (7)
  - More than $200,000 (8)
  - Don't know/ Prefer not to answer (9)

### `cps21_income_cat`
- Label: Nous n'avons pas besoin du montant exact. Le revenu de votre
- Question: ménage se situe-t-il dans l'une des catégories suivantes?
- Options:
  - Aucun revenu (1)
  - 1$ à 30 000$ (2)
  - 30 001$ à 60 000$ (3)
  - 60 001$ à 90 000$ (4)
  - 90 001$ à 110 000$ (5)
  - 110 001$ à 150 000$ (6)
  - 150 001$ à 200 000$ (7)
  - Plus de 200 000$ (8)
  - Je ne sais pas / Préfère ne pas répondre (9)

### `cps21_yob_2`
- Label: In what year were you born?
- Options:
  - ▼ 2003 (1) ... 1920 (84)

### `cps21_yob_2`
- Label: En quelle année êtes-vous né(e)?
- Options:
  - ▼ 2003 (1) ... 1920 (84)

### `cps21_property`
- Label: Do you or a member of your household own a residence (for example,
- Question: a home or an apartment), own a business (for example, a piece of property, a farm, or livestock), have stocks or bonds, or have savings? Please check all that apply. ▢ Own a residence (cps21_property_1) ▢ Own a business, a piece of property, a farm or livestock

### `cps21_property`
- Label: Est-ce que vous ou un membre de votre ménage possédez une
- Question: résidence (par exemple, une maison ou un appartement), une entreprise (par exemple, une propriété, une ferme ou un élevage), avez des actions ou des obligations, ou des économies? Veuillez sélectionner toutes les options qui s'appliquent. ▢ Possède une résidence (cps21_property_1)

### `cps21_marital`
- Label: Are you presently married, living with a partner, divorced, separated,
- Question: widowed, or have you never been married?
- Options:
  - Married (1)
  - Living with a partner (2)
  - Divorced (3)
  - Separated (4)
  - Widowed (5)
  - Never Married (6)
  - Don't know/ Prefer not to answer (7)

### `cps21_marital`
- Label: Êtes-vous présentement marié(e), vivant avec un(e) conjoint(e),
- Question: divorcé(e), séparé(e), veuf (veuve), ou n'avez-vous jamais été marié(e)?
- Options:
  - Marié(e) (1)
  - Vivant(e) avec un conjoint(e) de fait (2)
  - Divorcé(e) (3)
  - Séparé(e) (4)
  - Veuf/veuve (5)
  - Jamais marié(e) (6)
  - Je ne sais pas / Préfère ne pas répondre (7)

### `cps21_household`
- Label: Counting yourself, how many people live in your household?
- Question: ________________________________________________________________

### `cps21_household`
- Label: En vous incluant, combien de personnes votre ménage comporte-
- Question: t-il? ________________________________________________________________ Election Riding Information feduid Federal electoral district unique identifier
- Options:
  - • Not flagged (no errors flagged) (0)
  - • Flagged with error (1)
  - Clean complete (0)
  - Duplicate PID (1)
  - Duplicate IP & Demo (2)
  - Inattentive (4)

### `pes21_StartDate`
- Label: Start timestamp for the Post Election Survey response

### `pes21_StartDate_DMY`
- Label: Start timestamp for the Post Election Survey response in Day-
- Question: Month_year format

### `pes21_EndDate`
- Label: End timestamp for the Post Election Survey response

### `pes21_time`
- Label: How long the respondent spent in the Post Election Survey, in minutes
- Question: wave Flags respondents that completed the PES. “recontact” indicates the respondent completed the PES. Post-Election Suvey Questions Consent documentation

## Post-Election Survey (PES)

### `pes21_consent`
- Label: Letter of Information and Consent
- Question: Ontario, (519) 661-2111x85164, laura.stephenson@uwo.ca Montréal, (514) 987-3000 x5676, harell.allison@uqam.ca Peter Loewen, PhD, Political Science, Munk School of Global Affairs, University of Toronto, (416) 978-5120, peter.loewen@utoronto.ca
- Options:
  - I consent to participate in this study. I have read all of the information about the study. (1)
  - I do not consent to participate. (2)

### `pes21_consent`
- Label: Lettre d'information et de consentement
- Question: Titre du projet: Étude électorale canadienne 2021 Chercheure principale: Laura Stephenson, PhD, Science politique, University of Western Ontario, (519) 661 2111x85164, laura.stephenson@uwo.ca Cochercheurs: Allison Harell, PhD, Science politique, Université de Québec à Montréal,
- Options:
  - Daniel Rubenson, PhD, Politique et administration publique, Ryerson University, (416)
  - Je consens à participer à cette étude. J'ai lu toutes les informations à propos de l'étude. (1)
  - Je ne consens pas à participer. (2)

### `pes21_province`
- Label: In which province or territory are you currently living?
- Options:
  - Alberta (1)
  - British Columbia (2)
  - Manitoba (3)
  - New Brunswick (4)
  - Newfoundland and Labrador (5)
  - Northwest Territories (6)
  - Nova Scotia (7)
  - Nunavut (8)
  - Ontario (9)
  - Prince Edward Island (10)
  - Quebec (11)
  - Saskatchewan (12)
  - Yukon (13)

### `pes21_province`
- Label: Dans quelle province ou territoire habitez-vous présentement?
- Question: Main issue in election campaign
- Options:
  - Alberta (1)
  - Colombie-Britannique (2)
  - Manitoba (3)
  - Nouveau-Brunswick (4)
  - Terre-Neuve-et-Labrador (5)
  - Territoires du Nord-Ouest (6)
  - Nouvelle-Écosse (7)
  - Nunavut (8)
  - Ontario (9)
  - Île-du-Prince-Édouard (10)
  - Québec (11)
  - Saskatchewan (12)
  - Yukon (13)

### `pes21_mostimpissue`
- Label: Now we'd like to ask you some questions about the recent
- Question: federal election. What was the main issue in the campaign? ________________________________________________________________

### `pes21_mostimpissue`
- Label: Maintenant, nous aimerions vous poser quelques questions sur
- Question: l'élection fédérale qui a eu lieu récemment. Quel était l'enjeu principal de la campagne? Si vous ne le savez pas ou si vous préférez ne pas répondre, veuillez appuyer sur → ________________________________________________________________

### `pes21_partyissue`
- Label: Now thinking about each party, what issue or message did they
- Question: focus on the most during the campaign?
- Options:
  - Liberal Party (pes21_partyissue_4) ____________________________
  - Conservative Party (pes21_partyissue_5) _______________________
  - NDP (pes21_partyissue_6) __________________________________
  - BQ (pes21_partyissue_7) ____________________________________
  - Green Party (pes21_partyissue_8) _____________________________
  - People's Party (pes21_partyissue_9) ___________________________

### `pes21_partyissue`
- Label: Si l'on considère maintenant chaque parti politique, sur quel sujet ou
- Question: message se sont-ils concentrés le plus pendant la campagne électorale ? Si vous ne le savez pas ou si vous préférez ne pas répondre, veuillez appuyer sur → response option in pes21_partyissue. Turnout and vote choice in 2021 federal election
- Options:
  - Parti libéral (pes21_partyissue_4) _______________________
  - Parti conservateur (pes21_partyissue_5) __________________
  - NPD (pes21_partyissue_6) _____________________________
  - Bloc québécois (pes21_partyissue_7) _____________________
  - Parti vert (pes21_partyissue_8) __________________________
  - Parti populaire (pes21_partyissue_9) ______________________

### `pes21_turnout2021`
- Label: The federal election was held on Monday, September 20. In any
- Question: election, some people are not able to vote because they are sick or busy, or for some other reason. Others do not want to vote. Did you vote in the recent federal election?
- Options:
  - Yes (1)
  - No (2)
  - I usually vote but didn't this time (3)
  - I thought about voting but didn't (4)
  - I wasn’t registered to vote (5)
  - I was not eligible to vote (8)
  - Don't know/ Don't remember (6)
  - Prefer not to answer (7)

### `pes21_turnout2021`
- Label: Les élections fédérales ont eu lieu le lundi 20 septembre. Dans
- Question: toute élection, certaines personnes sont dans l’incapacité de voter parce qu’elles sont malades ou occupées, ou pour toute autre raison. D'autres personnes ne veulent pas voter. Avez-vous voté lors de l'élection fédérale la plus récente? Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je vote habituellement mais je ne l'ai pas fait cette fois-ci (3)
  - J'ai pensé voter mais je ne l'ai pas fait (4)
  - Je ne suis pas enregistré(e) pour voter (5)
  - Je ne suis pas éligible (8)
  - Je ne sais pas / Je ne m'en souviens pas (6)
  - Préfère ne pas répondre (7)

### `pes21_notvotereason1`
- Label: What is the main reason you did not vote?
- Options:
  - My vote will not make a difference, nothing will change, etc. (1)
  - No time, too busy, etc. (2)
  - No interest, did not follow election or issues, etc. (3)
  - Physical limitations, mobility issues, sick/ ill, aged, etc. (4)
  - Not able to prove ID or address (5)
  - Did not know when/ where to vote (6)
  - My requested mail ballot did not arrive in time (7)
  - I was isolating or quarantining related to COVID-19 (8)
  - The line to vote was too long (9)
  - Don't know/ Prefer not to answer (10)

### `pes21_notvotereason1`
- Label: Quelle est la raison principale qui explique pourquoi vous
- Question: n'avez pas voté? Display This Question: people are not able... = Yes
- Options:
  - Mon vote ne fera aucune différence, rien ne va changer (1)
  - Pas le temps, trop occupé(e) (2)
  - Pas d'intérêt, je n'ai pas suivi les élections ou les enjeux (3)
  - Limitations physiques, problèmes de mobilité, malade, âge (4)
  - Impossible de valider mon identité ou mon adresse (5)
  - Je ne savais pas quand / où voter (6)
  - Mon bulletin n'a pas arrivé à temps par le poste (7)
  - J'étais en isolement ou quarantaine lié à la COVID-19 (8)
  - La fille d'attente a été trop longue (9)
  - Je ne sais pas / Préfère ne pas répondre (10)

### `pes21_howvote`
- Label: There are many options for people wanting to vote in an election. How
- Question: did you vote?
- Options:
  - At a polling station on election day, September 20 (1)
  - At an advance polling station (or advance polls) (2)
  - At a local Elections Canada office (3)
  - By mail (4)
  - At home (5)
  - On campus (6)
  - Other (please specify) (7): pes21_howvote_7_TEXT
  - Don't know/ Prefer not to answer (8)

### `pes21_howvote`
- Label: Il existe de nombreuses options pour les personnes souhaitant voter
- Question: lors d'une élection. Comment avez-vous voté? Display This Question: vote? = At a polling station on election day, September 20 Or There are many options for people wanting to vote in an election. How did you
- Options:
  - Le jour du scrutin, le 20 septembre, au bureau de vote (1)
  - Au bureau de vote par anticipation (vote par anticipation) (2)
  - Dans un bureau local d'Élections Canada (3)
  - Par la poste (4)
  - À la maison (5)
  - Sur mon campus (6)
  - Autre (veuillez préciser) (7): pes21_howvote_7_TEXT
  - Je ne sais pas / Préfère ne pas répondre (8)

### `pes21_votingsafe`
- Label: How safe did you feel voting in person?
- Options:
  - Very safe (1)
  - Somewhat safe (2)
  - Not very safe (3)
  - Not safe at all (4)
  - Unsure (5)
  - Prefer not to answer (6)

### `pes21_votingsafe`
- Label: À quel point vous sentiez-vous en sécurité lorsque vous avez voté
- Question: en personne ? Display This Question: vote? = By mail pes_maileasy Did you find the process of requesting a mail-in ballot and voting by mail
- Options:
  - Très en sécurité (1)
  - Plutôt en sécurité (2)
  - Pas vraiment en sécurité (3)
  - Pas du tout en sécurité (4)
  - Je ne me souviens pas (5)
  - Préfère ne pas répondre (6)
  - Very easy (1)
  - Somewhat easy (2)
  - Somewhat difficult (3)
  - Very difficult (4)
  - Unsure (5)
  - Prefer not to answer (6)
  - Très facile (1)
  - Plutôt facile (2)
  - Plutôt difficile (3)
  - Très difficile (4)
  - Je ne suis pas certain(e) (5)
  - Préfère ne pas répondre (6)
  - Very easy (1)
  - Somewhat easy (2)
  - Somewhat difficult (3)
  - Very difficult (4)
  - Not sure (5)
  - Prefer not to answer (6)
  - Très facile (1)
  - Plutôt facile (2)
  - Plutôt difficile (3)
  - Très difficile (4)
  - Je ne suis pas certain(e) (5)
  - Préfère ne pas répondre (6)

### `pes21_votechoice2021`
- Label: Which party did you vote for?
- Question: Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - People's Party (6)
  - Another party (specify) (7): pes21_votechoice2021_7_TEXT
  - I spoiled my vote (8)
  - Don't know / Prefer not to answer (9)

### `pes21_votechoice2021`
- Label: Pour quel parti avez-vous voté?
- Question: Display This Choice: each response option in pes21_votechoice2021. Display This Question: Or Which party did you vote for? = Conservative Party
- Options:
  - Le Parti libéral (1)
  - Le Parti conservateur (2)
  - Le NPD (3)
  - Le Bloc québécois (4)
  - Le Parti vert (5)
  - Le Parti populaire (6)
  - Un autre parti (spécifiez) (7): pes21_votechoice2021_7_TEXT
  - J'ai annulé mon vote (8)
  - Je ne sais pas / Préfère ne pas répondre (9)

### `pes21_resason_chose`
- Label: What was the most important reason for choosing this party?
- Options:
  - Party policies (1)
  - Leader (2)
  - Local candidate (3)
  - Didn't want another party to win (4)
  - Other (5): pes21_resason_chose_5_TEXT
  - Don't know/ Prefer not to answer (6)

### `pes21_resason_chose`
- Label: Quelle a été la raison la plus importante pour laquelle vous
- Question: avez choisi ce parti? Display This Question: Or Which party did you vote for? = Conservative Party Or Which party did you vote for? = NDP
- Options:
  - Les politiques du parti (1)
  - Chef(fe) du parti (2)
  - Candidat(e) local(e) (3)
  - Je ne voulais pas qu'un autre parti gagne (4)
  - Autre (5): pes21_resason_chose_5_TEXT
  - Je ne sais pas / Préfère ne pas répondre (6)

### `pes21_when_decide`
- Label: When did you decide that you were going to vote for this party?
- Options:
  - Before the campaign (1)
  - During the campaign (2)
  - On election day (3)
  - I don't remember (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_when_decide`
- Label: Quand avez-vous décidé que vous alliez voter pour ce parti?
- Question: Display This Question: people are not able... = I wasn’t registered to vote Or The federal election was held on Monday, September 20. In any election, some people are not able... = I was not eligible to vote
- Options:
  - Avant la campagne (1)
  - Pendant la campagne (2)
  - Le jour des élections (3)
  - Je ne me souviens pas (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_pr_votechoice`
- Label: If you could have voted in the federal election on Sept. 20,
- Question: which party would you have voted for? Display This Choice:
- Options:
  - Liberal Party (1)
  - Conservative Party (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green Party (5)
  - People's Party (6)
  - Another party (specify) (7): pes21_pr_votechoice_7_TEXT
  - I would have spoiled my vote (8)
  - Don't know / Prefer not to answer (9)

### `pes21_pr_votechoice`
- Label: Si vous auriez pu voter lors de l'élection fédérale du 20
- Question: septembre, pour quel parti auriez-vous voté? Display This Choice: each response option in pes21_pr_votechoice. Satisfaction with democracy
- Options:
  - Le Parti libéral (1)
  - Le Parti conservateur (2)
  - Le NPD (3)
  - Le Bloc québécois (4)
  - Le Parti vert (5)
  - Le Parti populaire (6)
  - Un autre parti (spécifiez) (7): pes21_pr_votechoice_7_TEXT
  - J'aurais annulé mon vote (8)
  - Je ne sais pas / Préfère ne pas répondre (9)

### `pes21_dem_sat`
- Label: On the whole, are you very satisfied, fairly satisfied, not very satisfied,
- Question: or not satisfied at all with the way democracy works in Canada?
- Options:
  - Very satisfied (1)
  - Fairly satisfied (2)
  - Not very satisfied (3)
  - Not satisfied at all (4)
  - Don't know / Prefer not to answer (5)

### `pes21_dem_sat`
- Label: Dans l’ensemble, êtes-vous très satisfait, assez satisfait, pas très
- Question: satisfait ou pas du tout satisfait du fonctionnement de la démocratie au Canada? Campaign attention and party contact
- Options:
  - Très satisfait (1)
  - Plutôt satisfait (2)
  - Pas très satisfait (3)
  - Pas du tout satisfait (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_campatt`
- Label: How much attention did you pay to the election campaign?
- Options:
  - A lot (1)
  - Some (2)
  - Not much at all (3)
  - Don't know/ Prefer not to answer (4)

### `pes21_campatt`
- Label: À quel point avez-vous portée attention à la campagne électorale?
- Options:
  - Beaucoup (1)
  - Un peu (2)
  - Pas beaucoup (3)
  - Je ne sais pas / Préfère ne pas répondre (4)

### `pes21_where_info`
- Label: Where did you get most of your information about the campaign?
- Question: ________________________________________________________________

### `pes21_where_info`
- Label: Où avez-vous obtenu la plupart de vos informations sur la
- Question: campagne? Si vous ne le savez pas ou si vous préférez ne pas répondre, veuillez appuyer sur → ________________________________________________________________

### `pes21_contact1`
- Label: During the campaign, did a party or candidate contact you in person or
- Question: by any other means?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_contact1`
- Label: Durant la campagne, est-ce qu’un parti ou un candidat vous a
- Question: contacté en personne ou de toute autre façon? Display This Question: other means? = Yes
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_contact2`
- Label: Do you recall which parties or candidates contacted you? (Please
- Question: select all that apply) ▢ Liberal Party (pes21_contact2_1) ▢ Conservative Party (pes21_contact2_2) ▢ NDP (pes21_contact2_3)

### `pes21_contact2_7_TEXT`
- Label: ▢ Don’t know/ Prefer not to answer (pes21_contact2_8)

### `pes21_contact2`
- Label: Durant la campagne électorale, quel parti ou candidat vous a
- Question: contacté? (Veuillez sélectionner tous ceux qui s'appliquent) ▢ Parti libéral (pes21_contact2_1) ▢ Parti conservateur (pes21_contact2_2) ▢ Nouveau Parti démocratique (pes21_contact2_3)

### `pes21_contact2_7_TEXT`
- Label: ▢ Je ne sais pas / Préfère ne pas répondre (pes21_contact2_8)
- Question: response option in pes21_contact2. Government formation

### `pes21_formgovt`
- Label: Which should be more important to forming the government in
- Question: Canada:
- Options:
  - Winning the most seats (1)
  - Winning the most votes (2)
  - Don’t know/ Prefer not to answer (3)

### `pes21_formgovt`
- Label: Qu'est-ce qui devrait être le plus important pour former le
- Question: gouvernement au Canada: Party promises
- Options:
  - Gagner le plus de sièges (1)
  - Gagner le plus de votes (2)
  - Je ne sais pas/Préfère ne pas répondre (3)

### `pes21_keepromises`
- Label: Do political parties keep their election promises?
- Options:
  - Most of the time (1)
  - Some of the time (2)
  - Hardly ever (3)
  - Never (4)
  - Depends on the party (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_keepromises`
- Label: Les partis politiques tiennent-ils leurs promesses électorales:
- Question: Groups thermometers
- Options:
  - La plupart du temps (1)
  - Une partie du temps (2)
  - Presque jamais (3)
  - Jamais (4)
  - Cela dépend du parti (5)
  - Je ne sais pas / Préfère ne pas répondre (6)

### `pes21_groups1_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `pes21_groups1`
- Label: How do you feel about the following groups in Canada? Set the slider
- Question: to any number from 0 to 100, where 0 means you really dislike the group and 100 means you really like the group. Really dislike Really like Don't know/ Prefer not to

### `pes21_groups1`
- Label: Que pensez-vous des différents groupes ci-dessous? Veuillez glisser
- Question: la barre sur un chiffre entre 0 et 100, où 0 indique que vous n'aimez vraiment pas du tout le groupe et 100 que vous aimez vraiment beaucoup ce groupe. Si vous ne savez pas, ou préférez ne pas répondre, veuillez sélectionner → Je n'aime J'aime Je ne sais

### `pes21_paymed`
- Label: People who are willing to pay should be allowed to get medical
- Question: treatment sooner.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_paymed`
- Label: Les personnes disposées à payer devraient pouvoir obtenir des
- Question: traitements médicaux plus tôt.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_senate`
- Label: The Senate should be abolished.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_senate`
- Label: Le Sénat devrait être aboli.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_losetouch`
- Label: Those elected to Parliament soon lose touch with the people.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_losetouch`
- Label: Ceux qui sont élus au Parlement perdent vite contact avec la
- Question: population.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_hatespeech`
- Label: It should be illegal to say hateful things publicly about racial, ethnic
- Question: and religious groups.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_hatespeech`
- Label: Tenir des propos haineux à l'égard de groupes raciaux, ethniques
- Question: et religieux devrait être illégal. QID111 Timing
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `pes21_envirojob`
- Label: When there is a conflict between protecting the environment and
- Question: creating jobs, jobs should come first.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_envirojob`
- Label: Lorsqu'il existe un conflit entre la protection de l'environnement et la
- Question: création d'emplois, les emplois devraient avoir la priorité. Efficacy
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_govtcare`
- Label: The government does not care much about what people like me think.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_govtcare`
- Label: Le gouvernement ne se soucie pas beaucoup de ce que les gens
- Question: comme moi pensent. Family values
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_famvalues`
- Label: This country would have many fewer problems if there was more
- Question: emphasis on traditional family values.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_famvalues`
- Label: Il y aurait moins de problèmes dans ce pays si on accordait plus
- Question: d’importance aux valeurs familiales traditionnelles. randomized. The variable FL_327_DO_DemocracyCheckup_effic gives the display order for pes21_govtcare, and the variable FL_327_DO_DemocracyCheckup_effi0 gives the display order for pes21_famvalues.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_bilingualism`
- Label: We have gone too far in pushing bilingualism in Canada.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_bilingualism`
- Label: Nous sommes allés trop loin dans la promotion du bilinguisme au
- Question: Canada. Equal rights
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_equalrights`
- Label: We have gone too far in pushing equal rights in this country.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_equalrights`
- Label: On est allé trop loin dans la promotion des droits égaux dans ce
- Question: pays. randomized. The variable FL_49_DO_DemocracyCheckup_attitu gives the display order for pes21_bilingualism, and the variable FL_49_DO_DemocracyCheckup_democr gives the display order for
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_fitin`
- Label: Too many recent immigrants just don't want to fit in to Canadian society.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_fitin`
- Label: Un trop grand nombre d’immigrants récents ne veulent tout simplement pas
- Question: s'intégrer.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_immigjobs`
- Label: Immigrants take jobs away from other Canadians.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_immigjobs`
- Label: Les immigrants enlèvent des emplois aux autres Canadiens.
- Question: The variable FL_244_DO_DemocracyCheckup_attit gives the display order for pes21_fitin, and the variable FL_244_DO_DemocracyCheckup_atti0 gives the display order for pes21_immigjobs. Indigenous Resentment Scale
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_ab_favors`
- Label: Irish, Italian, Jewish and many other minorities overcame prejudice
- Question: and worked their way up. Aboriginal peoples in Canada should do the same without any special favors.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_ab_favors`
- Label: Les irlandais, les italiens, les juifs et plusieurs autres minorités ont su
- Question: surmonter les préjugés et réussir. Les peuples autochtones au Canada devraient pouvoir en faire autant sans recevoir un traitement de faveur.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_ab_deserve`
- Label: Over the past few years, Aboriginal peoples have gotten less than
- Question: they deserve.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_ab_deserve`
- Label: Les peuples autochtones ont reçu moins que ce qu'ils méritent au
- Question: cours des dernières années.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_ab_col`
- Label: Generations of colonialism and discrimination have created conditions
- Question: that make it difficult for Aboriginal peoples to work their way out of the lower class.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_ab_col`
- Label: Les conditions sociales et économiques sont telles qu'il est à peu près
- Question: impossible pour la plupart des peuples autochtones de sortir de la pauvreté. Government effectiveness
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_govtprograms`
- Label: Government can no longer ${e://Field/govt_programs_word} the
- Question: kinds of programs and services people want.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_govtprograms`
- Label: Le gouvernement ${e://Field/govt_programs_word_fr} les
- Question: programmes et les services que les gens désirent. Ties with foreign countries
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_tieus`
- Label: Do you think Canada's ties with the United States should be...
- Options:
  - Much closer (1)
  - Somewhat closer (2)
  - About the same as now (3)
  - Somewhat more distant (4)
  - Much more distant (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_tieus`
- Label: Croyez-vous que les liens entre le Canada et les États-Unis devraient
- Question: être...
- Options:
  - Beaucoup plus serrés (1)
  - Assez serrés (2)
  - A peu près les mêmes que maintenant (3)
  - Un peu plus distants (4)
  - Beaucoup plus distants (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_tiechina`
- Label: Do you think Canada's ties with China should be...
- Options:
  - Much closer (1)
  - Somewhat closer (2)
  - About the same as now (3)
  - Somewhat more distant (4)
  - Much more distant (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_tiechina`
- Label: Croyez-vous que les liens entre le Canada et la Chine devraient être...
- Question: The variable FL_346_DO_DemocracyCheckup_tiesw gives the display order for pes21_tieus, and the variable FL_346_DO_DemocracyCheckup_ties0 gives the display order for pes21_tiechina. Group identity
- Options:
  - Beaucoup plus serrés (1)
  - Assez serrés (2)
  - Un peu près les mêmes que maintenant (3)
  - Un peu plus distants (4)
  - Beaucoup plus distants (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_ethid`
- Label: How important are the following things to your identity?
- Question: Don't Not Not very Fairly Very know/ important important important important Prefer not at all (1) (2) (3) (4) to answer

### `pes21_ethid`
- Label: Quel est le niveau d'importance de chacune des caractéristiques
- Question: suivantes pour votre identité? Ne sais Pas du tout Pas très Assez Très pas/Préfère important important important important ne pas

### `pes21_can_id`
- Label: Some people say that the following things are important for being truly
- Question: Canadian. Others say they are not important. How important do you think the following is for being truly Canadian... Don't know Not Not very Fairly Very
- Options:
  - at all (1) (2) (3) (4)
  - o o o o

### `pes21_can_id`
- Label: Certaines personnes disent que les critères qui suivent sont importants
- Question: pour être véritablement Canadien(ne). D’autres disent qu’ils ne sont pas importants. Selon vous, quel est le niveau d’importance de ces critères pour être véritablement canadien(ne)… Je ne sais
- Options:
  - (1) (2) (3) (4)
  - o o o o
  - o o o o
  - o o o o
  - o o o o

### `pes21_conf_inst1`
- Label: Please indicate how much confidence you have in the following:
- Question: Don't know/ A great Quite a lot Not very None at Prefer not
- Options:
  - deal (1) (2) much (3) all (4)
  - o o o o

### `pes21_conf_inst1`
- Label: Quelle confiance accordez-vous aux institutions suivantes:
- Question: Je ne sais Beaucoup Pas du pas/ Préfère (1) tout (4) ne pas Le gouvernement
- Options:
  - Assez (2) Peu (3)
  - répondre (5)
  - o o o o

### `pes21_conf_inst2`
- Label: Please indicate how much confidence you have in the following:
- Question: Don't know/ A great Quite a lot Not very None at Prefer not
- Options:
  - deal (1) (2) much (3) all (4)

### `pes21_conf_inst2`
- Label: Quelle confiance accordez-vous aux institutions suivantes:
- Question: Je ne sais pas/ Beaucoup Pas du Préfère ne (1) tout (4) pas
- Options:
  - Assez (2) Peu (3)

### `pes21_emb_none`
- Label: Election ballots should have the option 'None of the above' for those
- Question: who do not support any of the candidates.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_emb_none`
- Label: Les bulletins de vote devraient inclure l'option "Aucun de ces
- Question: candidats" pour ceux et celles qui n'appuient aucun des candidats.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_lowturnout`
- Label: Low voter turnout weakens Canadian democracy.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_lowturnout`
- Label: Le faible taux de participation affaiblit la démocratie canadienne.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_internetvote1`
- Label: Canadians should have the option to vote over the Internet in
- Question: federal elections.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_internetvote1`
- Label: Les Canadiens devraient pouvoir voter par Internet lors des
- Question: élections fédérales.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_internetvote2`
- Label: If you could vote over the internet, how likely would you be to do
- Question: so?
- Options:
  - Very likely (1)
  - Somewhat likely (2)
  - Not very likely (3)
  - Not at all likely (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_internetvote2`
- Label: Si vous pouviez voter sur Internet, quelle serait la probabilité que
- Question: vous le fassiez?
- Options:
  - Très probable (1)
  - Assez probable (2)
  - Peu probable (3)
  - Improbable (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_foreign`
- Label: How confident are you that our elections are safe from foreign
- Question: interference?
- Options:
  - Very confident (1)
  - Somewhat confident (2)
  - Not very confident (3)
  - Not at all confident (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_foreign`
- Label: À quel point êtes-vous confiant que nos élections sont à l'abri de
- Question: l'interférence étrangère?
- Options:
  - Très confiant (1)
  - Assez confiant (2)
  - Pas très confiant (3)
  - Pas du tout confiant (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_emb_satif`
- Label: How satisfied are you with the way Elections Canada runs federal
- Question: elections?
- Options:
  - Very satisfied (1)
  - Fairly satisfied (2)
  - Not very satisfied (3)
  - Not satisfied at all (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_emb_satif`
- Label: À quel point êtes-vous satisfait de la façon dont Élections Canada a
- Question: pris en charge les élections fédérales?
- Options:
  - Très satisfait (1)
  - Assez satisfait (2)
  - Pas très satisfait (3)
  - Pas du tout satisfait (4)
  - Je ne sais pas/ Préfère ne pas répondre (5)

### `pes21_emb8`
- Label: Thinking about this election, would you say that Elections Canada ran the
- Question: election…
- Options:
  - Very fairly (1)
  - Somewhat fairly (2)
  - Not very fairly (3)
  - Not at all fairly (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_emb8`
- Label: En pensant à cette élection, croyez-vous qu'Élections Canada a conduit
- Question: l'élection…
- Options:
  - De façon très juste (1)
  - De façon assez juste (2)
  - Pas très justement (3)
  - Pas justement du tout (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_internetregis`
- Label: To complete voter registration online, electors must provide their
- Question: date of birth, home address, and driver’s license number on Elections Canada’s website. How comfortable are you with providing this information online to Elections Canada?
- Options:
  - Very comfortable (1)
  - Somewhat comfortable (2)
  - Not very comfortable (3)
  - Not at all comfortable (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_internetregis`
- Label: Pour compléter leur inscription en ligne, les électeurs devraient
- Question: fournir leur date de naissance, leur adresse et leur numéro de permis de conduire sur le site d'Élections Canada. À quel point seriez vous à l'aise de fournir ces informations en ligne à Élections Canada?
- Options:
  - Très à l'aise (1)
  - Un peu à l'aise (2)
  - Pas très à l'aise (3)
  - Pas du tout à l'aise (4)
  - Ne sais pas / Préfère ne pas répondre (5)

### `pes21_internetrisk1`
- Label: Which statement comes closest to your own view?
- Options:
  - Voting on the Internet is risky. (1)
  - Voting on the Internet is safe. (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_internetrisk1`
- Label: Quelle affirmation reflète le mieux votre opinion?
- Options:
  - Voter sur Internet est risqué (1)
  - Voter sur Internet est sécuritaire (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_internetrisk2`
- Label: Which statement comes closest to your own view?
- Options:
  - Registering to vote on the Internet is risky. (1)
  - Registering to vote on the Internet is safe. (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_internetrisk2`
- Label: Quelle affirmation reflète le mieux votre opinion?
- Options:
  - S'enregistrer sur Internet pour voter est risqué (1)
  - S'enregistrer sur Internet pour voter est sécuritaire (2)
  - Je ne sais pas / Préfère ne pas répondre (3)

### `pes21_emb_register`
- Label: Did you receive a voter registration card in the mail?
- Options:
  - Yes (1)
  - No (2)
  - Don't know (3)
  - Prefer not to answer (4)

### `pes21_emb_register`
- Label: Avez-vous reçu une carte d'information de l'électeur par la poste?
- Question: Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas (3)
  - Préfère ne pas répondre (4)

### `pes21_emb_card`
- Label: Was the information on your voter registration card correct?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_emb_card`
- Label: Les informations sur votre carte d'information de l'électeur étaient-
- Question: elles correctes? Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_emb_register2`
- Label: Did you register to vote during the election?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_emb_register2`
- Label: Vous êtes-vous inscrit(e) lors de l'élection pour pouvoir voter?
- Question: Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_emb_reg_how`
- Label: How did you register to vote?
- Options:
  - Online through the Elections Canada website (1)
  - At my local Elections Canada office (2)
  - At the polls (3)
  - By mail (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_emb_reg_how`
- Label: Comment vous êtes-vous inscrit(e) lors de l'élection pour
- Question: pouvoir voter? Display This Question:
- Options:
  - En ligne sur le site d'Élections Canada (1)
  - À un bureau local d'Élections Canada (2)
  - Aux urnes (3)
  - Par la poste (4)
  - Je ne sais pas/ Préfère ne pas répondre (5)

### `pes21_emb_register3`
- Label: How easy or difficult did you find it to register to vote?
- Options:
  - Very easy (1)
  - Somewhat easy (2)
  - Neither easy nor difficult (3)
  - Somewhat difficult (4)
  - Very difficult (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_emb_register3`
- Label: À quel point a-t-il été facile de vous inscrire pour voter?
- Options:
  - Très facile (1)
  - Plutôt facile (2)
  - Ni facile, ni difficile (3)
  - Plutôt difficile (4)
  - Très difficile (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_emb4`
- Label: From which of the following sources did you get information about voting
- Question: in the federal election? (Select all that apply) ▢ Elections Canada flyer (pes21_emb4_1) ▢ Voter Information Card (pes21_emb4_2) ▢ Advertisements from Elections Canada (pes21_emb4_3)

### `pes21_emb4`
- Label: De quelles sources avez-vous reçu des informations sur le vote aux
- Question: élections fédérales? (Séléctionnez toutes les sources qui s'appliquent) ▢ Bulletin d'Élections Canada (pes21_emb4_1) ▢ Carte d'information de l'électeur (pes21_emb4_2) ▢ Publicités d'Élections Canada (pes21_emb4_3)

### `pes21_emb7`
- Label: Thinking about the recent election, how informed did you feel about the
- Question: following elements of the voting process? Not informed at all Very informed 0 1 2 3 4 5 6 7 8 9 10 What documentation was required to

### `pes21_emb7`
- Label: En ce qui concerne les dernières élections, nous aimerions savoir à quel
- Question: point vous vous sentiez informé au sujet des différents éléments du processus électoral ci-dessous: Si vous ne savez pas ou si vous préférez ne pas répondre, veuillez appuyer sur → Pas du tout informé Très informé

### `pes21_emb_info`
- Label: How easy or difficult was it to find the information you needed to
- Question: vote?
- Options:
  - Extremely easy (1)
  - Somewhat easy (2)
  - Neither easy nor difficult (3)
  - Somewhat difficult (4)
  - Extremely difficult (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_emb_info`
- Label: À quel point était-il facile ou difficile de trouver les informations dont
- Question: vous aviez besoin pour voter?
- Options:
  - Très facile (1)
  - Plutôt facile (2)
  - Ni facile, ni difficile (3)
  - Plutôt difficile (4)
  - Très difficile (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_embsatisfy`
- Label: Overall, how satisfied were you with your voting experience?
- Options:
  - Very satisfied (1)
  - Somewhat satisfied (2)
  - Somewhat dissatisfied (3)
  - Very dissatisfied (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_embsatisfy`
- Label: Dans l'ensemble, dans quelle mesure avez-vous été satisfait(e) de
- Question: votre expérience de vote ? Provincial election
- Options:
  - Très satisfait(e) (1)
  - Plutôt satisfait(e) (2)
  - Plutôt insatisfait(e) (3)
  - Très insatisfait(e) (4)
  - Ne sais pas / Préfère ne pas répondre (5)

### `pes21_provvote`
- Label: If a provincial election were held today in
- Question: ${pes21_province/ChoiceGroup/SelectedChoices}, which party would you vote for? Display This Choice: Or In which province or territory are you currently living? = Saskatchewan Or In which province or territory are you currently living? = Manitoba
- Options:
  - Liberal (1)
  - NDP (2)
  - Green (3)
  - Coalition Avenir Québec (4)
  - Parti Québécois (5)
  - Québec Solidaire (6)
  - United Conservative (7)
  - Alberta Party (8)
  - Buffalo Party (9)
  - Saskatchewan Party (10)
  - Progressive Conservative (11)
  - People's Alliance (12)
  - Yukon Party (13)
  - Another party (please specify) (14): pes21_provvote_14_TEXT
  - None of these (15)
  - Don't know/ Prefer not to answer (16)

### `pes21_provvote`
- Label: En politique provinciale, vous considérez-vous habituellement
- Question: comme étant: Display This Choice: Or In which province or territory are you currently living? = Saskatchewan Or In which province or territory are you currently living? = Manitoba
- Options:
  - Parti libéral (1)
  - NPD (2)
  - Parti vert (3)
  - Coalition avenir Québec (4)
  - Parti québécois (5)
  - Québec solidaire (6)
  - United Conservative (7)
  - Alberta Party (8)
  - Parti conservateur (9)
  - Saskatchewan Party (10)
  - Parti progressiste-conservateur du Canada (11)
  - Alliance des gens du Nouveau-Brunswick (12)
  - Yukon Party (13)
  - Autre parti (veuillez spécifier) (14): pes21_provvote_14_TEXT
  - Aucun de ces partis (15)
  - Je ne sais pas/Préfère ne pas répondre (16)

### `pes21_friendsnames`
- Label: Can you give me the first names or initials of the three people
- Question: you talked with most about politics during the past year? These people might be from your family, from work, from the neighborhood, from church, from some other organization you belong to, or they might be from somewhere else. Please provide their first names only.
- Options:
  - (pes21_friendsnames_1)
  - (pes21_friendsnames_2)
  - (pes21_friendsnames_3)

### `pes21_friendsnames`
- Label: Pouvez-vous me donner les prénoms ou les initiales des trois
- Question: personnes avec lesquelles vous avez le plus parlé de politique au cours de la dernière année ? Il peut s'agir de personnes de votre famille, de votre travail, du quartier, de l'église, d'une autre organisation à laquelle vous appartenez, ou d'autres personnes. Veuillez indiquer uniquement leur prénom :
- Options:
  - (pes21_friendsnames_1)
  - (pes21_friendsnames_2)
  - (pes21_friendsnames_3)

### `pes21_friendswho`
- Label: Thinking
- Question: about ${e://Field/dc_social_network_name_1}${e://Field/dc_social_network_name_2} ${e://Field/dc_social_network_name_3} are they someone who… (select all that apply) Display This Answer: Display This Answer: Display This Answer: Text Response Is Not _2 Text Response Is Text Response Is Not
- Options:
  - } (1) alue/2} (2) } (3)
  - views (1)
  - you (3)

### `pes21_friendswho`
- Label: En pensant de ${e://Field/dc_social_network_name_1}
- Question: ${e://Field/dc_social_network_name_2}${e://Field/dc_social_network_name_3}, sont- elles des personnes qui... (séléctionnez tous les choix qui s'appliquent) Display This Answer: Display This Answer: Display This Answer: Text Response Is Not 2 Text Response Is Text Response Is Not
- Options:
  - } (1) /2} (2) } (3)
  - ethnicité (2)
  - vous (3)
  - n (5)

### `pes21_discfam`
- Label: How often do you discuss politics with family and friends?
- Options:
  - Never (1)
  - Sometimes (2)
  - Often (3)
  - All the time (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_discfam`
- Label: À quelle fréquence discutez-vous de politique avec votre famille et vos
- Question: amis? Political participation
- Options:
  - Jamais (1)
  - Parfois (2)
  - Souvent (3)
  - Tout le temps (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_partic1`
- Label: Here are some things people can do to participate in politics. Please
- Question: indicate how many times you've done these things over the past 12 months. Don't More than know/ Just once A few
- Options:
  - (2) times (3)
  - o o o o

### `pes21_partic1`
- Label: Voici certaines choses que les gens peuvent faire pour participer à la
- Question: vie politique. Veuillez indiquer combien de fois vous avez fait ces choses au cours des 12 derniers mois. Je ne sais pas /
- Options:
  - o o o o
  - o o o o

### `pes21_partic2`
- Label: Here are some things people can do to participate in politics. Please
- Question: indicate how many times you've done these things over the past 12 months. Don't More than know/ Just once A few
- Options:
  - (2) times (3)
  - o o o o
  - o o o o
  - o o o o

### `pes21_partic2`
- Label: Voici certaines choses que les gens peuvent faire pour participer à la
- Question: vie politique. Veuillez indiquer combien de fois vous avez fait ces choses au cours des 12 derniers mois. Je ne sais pas /
- Options:
  - o o o o
  - o o o o

### `pes21_partic3`
- Label: Here are some things people can do to participate in politics. Please
- Question: indicate how many times you've done these things over the past 12 months. Don't More than know/ Just once A few
- Options:
  - (2) times (3)
  - o o o o

### `pes21_partic3`
- Label: Voici certaines choses que les gens peuvent faire pour participer à la
- Question: vie politique. Veuillez indiquer combien de fois vous avez fait ces choses au cours des 12 derniers mois . Je ne sais pas Plus de

### `pes21_partymember`
- Label: Have you ever been a member of a provincial or federal political
- Question: party?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_partymember`
- Label: Avez-vous déjà été membre d'un parti politique provincial ou
- Question: fédéral? Representation by women
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_womenparl`
- Label: The best way to protect women's interests is to have more women
- Question: in Parliament.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_womenparl`
- Label: Le meilleur moyen de protéger les intérêts des femmes est d'avoir
- Question: plus de femmes au Parlement. Populism
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_populism_2`
- Label: What people call compromise in politics is really just selling out on
- Question: one's principles.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know / Prefer not to answer (6)

### `pes21_populism_2`
- Label: En politique, les compromis équivalent à renoncer à ses principes.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_populism_3`
- Label: Most politicians do not care about the people.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know / Prefer not to answer (6)

### `pes21_populism_3`
- Label: La plupart des politiciens ne se soucient pas du peuple.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_populism_4`
- Label: Most politicians are trustworthy.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know / Prefer not to answer (6)

### `pes21_populism_4`
- Label: La plupart des politiciens sont dignes de confiance.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_populism_6`
- Label: Having a strong leader in government is good for Canada even if
- Question: the leader bends the rules to get things done.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know / Prefer not to answer (6)

### `pes21_populism_6`
- Label: Avoir un leader fort à la tête du gouvernement est bon pour le
- Question: Canada même si ce leader contourne les règles pour faire avancer les choses.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_populism_7`
- Label: The people, and not politicians, should make our most
- Question: important policy decisions.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know / Prefer not to answer (6)

### `pes21_populism_7`
- Label: C'est le peuple, et non les politiciens, qui devrait prendre les
- Question: décisions politiques les plus importantes.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_populism_8`
- Label: Most politicians care only about the interests of the rich and
- Question: powerful.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know / Prefer not to answer (6)

### `pes21_populism_8`
- Label: La plupart des politiciens se soucient uniquement des intérêts des
- Question: gens riches et puissants. pes21_populism_3, pes21_populism_4, pes21_populism_6, pes21_populism_7, pes21_populism_8. The following variables give the display order for each variable:
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_donerm`
- Label: How much do you think should be done for racial minorities?
- Options:
  - Much more (1)
  - Somewhat more (2)
  - About the same as now (3)
  - Somewhat less (4)
  - Much less (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_donerm`
- Label: Selon vous, combien devrait être fait pour les minorités raciales?
- Options:
  - Beaucoup plus (1)
  - Un peu plus (2)
  - Ni plus ni moins (3)
  - Un peu moins (4)
  - Beaucoup moins (5)
  - Je ne sais pas / Préfère ne pas répondre (6)

### `pes21_donew`
- Label: How much do you think should be done for women?
- Options:
  - Much more (1)
  - Somewhat more (2)
  - About the same as now (3)
  - Somewhat less (4)
  - Much less (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_donew`
- Label: Selon vous, combien devrait être fait pour les femmes?
- Options:
  - Beaucoup plus (1)
  - Un peu plus (2)
  - Ni plus ni moins (3)
  - Un peu moins (4)
  - Beaucoup moins (5)
  - Je ne sais pas / Préfère ne pas répondre (6)

### `pes21_donegl`
- Label: How much do you think should be done for gays and lesbians?
- Options:
  - Much more (1)
  - Somewhat more (2)
  - About the same as now (3)
  - Somewhat less (4)
  - Much less (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_donegl`
- Label: Selon vous, combien devrait être fait pour les gais et lesbiennes?
- Options:
  - Beaucoup plus (1)
  - Un peu plus (2)
  - Ni plus ni moins (3)
  - Un peu moins (4)
  - Beaucoup moins (5)
  - Je ne sais pas / Préfère ne pas répondre (6)

### `pes21_doneqc`
- Label: How much do you think should be done for Quebec?
- Options:
  - Much more (1)
  - Somewhat more (2)
  - About the same as now (3)
  - Somewhat less (4)
  - Much less (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_doneqc`
- Label: Selon vous, combien devrait être fait pour le Québec?
- Question: • The variable FL_312_DO_Howmuchshouldbedonefor gives the display order for pes21_donerm. • The variable FL_312_DO_Howmuchshouldbedonefo0 gives the display order for pes21_donew.
- Options:
  - Beaucoup plus (1)
  - Un peu plus (2)
  - Ni plus ni moins (3)
  - Un peu moins (4)
  - Beaucoup moins (5)
  - Je ne sais pas / Préfère ne pas répondre (6)

### `pes21_abort2`
- Label: Should abortion be banned?
- Options:
  - Yes (1)
  - In some circumstances (2)
  - No (3)
  - Don't know/ Prefer not to answer (4)

### `pes21_abort2`
- Label: L'avortement devrait-il être interdit?
- Question: Conversion therapy
- Options:
  - Oui (1)
  - Dans certaines circonstances (2)
  - Non (3)
  - Je ne sais pas/ Préfère ne pas répondre (4)

### `pes21_conversion_the`
- Label: Conversion therapy is when mental health practitioners try to
- Question: change a LGBTQ person’s sexual orientation or gender identity. Do you think that conversion therapy should be legal or illegal to use on LGBTQ children under age 18?
- Options:
  - Legal (1)
  - Illegal (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_conversion_the`
- Label: La thérapie de conversion est lorsque les praticiens de la
- Question: santé mentale essaient de changer l'orientation sexuelle ou l'identité de genre d'une personne LGBTQ. Pensez-vous que la thérapie de conversion devrait être légale ou illégale à utiliser sur les enfants LGBTQ de moins de 18 ans ? Economic attitudes
- Options:
  - Légal (1)
  - Illégal (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_trade`
- Label: International trade creates more jobs in Canada than it destroys.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_trade`
- Label: Le commerce international crée plus d'emplois au Canada qu'il n'en
- Question: détruit.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_privjobs`
- Label: The government should leave it entirely to the private sector to create
- Question: jobs.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_privjobs`
- Label: Le gouvernement devrait laisser au secteur privé l'entière
- Question: responsabilité de créer des emplois.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_blame`
- Label: People who don't get ahead should blame themselves, not the system.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_blame`
- Label: Ceux qui ne réussissent pas dans la vie devraient se blâmer eux-
- Question: mêmes, pas le système.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt d'accord (4)
  - Fortement d'accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_stdofliving`
- Label: The government should:
- Options:
  - See to it that everyone has a decent standard of living (1)
  - Leave people to get ahead on their own (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_stdofliving`
- Label: Le gouvernement devrait:
- Question: Social Trust
- Options:
  - Voir à ce que tout le monde ait un niveau de vie décent (1)
  - Laisser les gens avancer par eux-mêmes (2)
  - Je ne sais pas / Préfère ne pas répondre (3)

### `pes21_trust`
- Label: Generally speaking would you say that most people can be trusted, or that
- Question: you need to be very careful when dealing with people?
- Options:
  - Most people can be trusted (1)
  - You need to be very careful when dealing with people (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_trust`
- Label: En général, diriez-vous qu'on peut faire confiance à la plupart des gens,
- Question: ou qu'on doit être très prudent dans nos relations avec les autres. Economic inequalities
- Options:
  - On peut faire confiance à la plupart des gens (1)
  - On devrait être très prudent dans nos relations avec les autres (2)
  - Je ne sais pas / Préfère ne pas répondre (3)

### `pes21_inequal`
- Label: Is income inequality a big problem in Canada?
- Options:
  - Definitely yes (1)
  - Probably yes (2)
  - Not sure (3)
  - Probably not (4)
  - Definitely not (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_inequal`
- Label: Est-ce que les inégalités de revenu sont un problème important au
- Question: Canada?
- Options:
  - Définitivement oui (1)
  - Probablement oui (2)
  - Pas certain(e) (3)
  - Probablement pas (4)
  - Définitivement pas (5)
  - Je ne sais pas / Préfère ne pas répondre (6)

### `pes21_gap`
- Label: How much do you think should be done to reduce the gap between the rich
- Question: and the poor in Canada?
- Options:
  - Much more (1)
  - Somewhat more (2)
  - About the same as now (3)
  - Somewhat less (4)
  - Much less (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_gap`
- Label: Que devrait-on faire pour réduire les écarts entre les riches et les pauvres
- Question: au Canada? Federal-provincial relations
- Options:
  - Beaucoup plus (1)
  - Un peu plus (2)
  - Ni plus ni moins (3)
  - Un peu moins (4)
  - Beaucoup moins (5)
  - Je ne sais pas / Préfère ne pas répondre (6)

### `pes21_prov_treatment`
- Label: In general, does the federal government treat your province...
- Options:
  - Better (1)
  - Worse (2)
  - The same as other provinces (3)
  - Don't know/ Prefer not to answer (4)

### `pes21_prov_treatment`
- Label: En général, le gouvernement fédéral traite-t-il votre province...
- Options:
  - Mieux (1)
  - Pire (2)
  - Similaire aux autres province (3)
  - Je ne sais pas/ Préfère ne pas répondre (4)

### `pes21_provfed`
- Label: Which do you prefer:
- Options:
  - A strong federal government (1)
  - More power to the provincial governments (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_provfed`
- Label: Lequel préfèrez-vous:
- Question: Issue positions
- Options:
  - Un gouvernement fédéral fort (1)
  - Plus de pouvoir accordé aux gouvernements provinciaux (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_hostile2`
- Label: Women seek to gain power by getting control over men.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_hostile2`
- Label: Les femmes cherchent à gagner du pouvoir en contrôlant les hommes.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_hostile4`
- Label: Women exaggerate problems they have at work.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_hostile4`
- Label: Les femmes exagèrent les problèmes qu’elles ont sur leur lieu de
- Question: travail. FL_149_DO_Other_statements_Sexis gives the display order for pes21_hostile2, and the variable FL_149_DO_Other_statements_Sexi0 gives the display order for pes21_hostile4.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Ne sais pas / Préfère ne pas répondre (6)

### `pes21_pos_carbon`
- Label: To help reduce greenhouse gas emissions, the federal
- Question: government should continue the carbon tax.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_pos_carbon`
- Label: Afin d'aider à réduire les émissions de gaz à effet de serre, le
- Question: gouvernement fédéral devrait maintenir la taxe sur le carbone.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_pos_energy`
- Label: The federal government should do more to help Canada’s energy
- Question: sector, including building oil pipelines.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_pos_energy`
- Label: Le gouvernement fédéral devrait en faire davantage afin d'aider le
- Question: secteur énergétique canadien, notamment en construisant des oléoducs. randomized. FL_146_DO_Issuepositions_carbont gives the display order for pes21_pos_carbon, and the variable FL_146_DO_Issuepositions_energys gives the display order for pes21_pos_energy.
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_cc1`
- Label: Do you think that climate change is happening?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_cc1`
- Label: Pensez-vous que les changements climatiques se produisent réellement?
- Question: Display This Question:
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_cc2`
- Label: What do you think is the main cause of climate change?
- Options:
  - Human activities such as burning fossil fuels for energy (1)
  - Natural changes in the environment (2)
  - Other (3): pes21_cc2_3_TEXT
  - Don't know/ Prefer not to answer (4)

### `pes21_cc2`
- Label: Quelle est la cause des changements climatiques selon vous?
- Question: Partisanship
- Options:
  - Causés principalement par des activités humaines telles que la combustion des énergies fossiles (1)
  - Causés principalement par des changements naturels dans l'environnement (2)
  - Autre (3): pes21_cc2_3_TEXT
  - Je ne sais pas/ Préfère ne pas répondre (4)

### `pes21_pidtrad_t`
- Label: Timing
- Options:
  - First Click (1)
  - Last Click (2)
  - Page Submit (3)
  - Click Count (4)

### `pes21_pidtrad`
- Label: In federal politics, do you usually think of yourself as a:
- Question: Display This Choice:
- Options:
  - Liberal (1)
  - Conservative (2)
  - NDP (3)
  - Bloc Québécois (4)
  - Green (5)
  - People’s Party (6)
  - Another party (please specify) (7): pes21_pidtrad_7_TEXT
  - None of these (8)
  - Don't know/ Prefer not to answer (9)

### `pes21_pidtrad`
- Label: En politique fédérale, vous considérez-vous habituellement comme
- Question: étant : Display This Choice: response option in pes21_pidtrad. Display This Question:
- Options:
  - Libéral (1)
  - Conservateur (2)
  - NPD (3)
  - Bloc québécois (4)
  - Parti vert (5)
  - Parti populaire du Canada (6)
  - Un autre parti (Veuillez spécifier) (7): pes21_pidtrad_7_TEXT
  - Aucun de ces partis (8)
  - Je ne sais pas/Préfère ne pas répondre (9)

### `pes21_pidtradstrong`
- Label: How
- Question: strongly ${e://Field/pid_en}${pes21_pidtrad/ChoiceTextEntryValue/7} do you feel?
- Options:
  - Very strongly (1)
  - Fairly strongly (2)
  - Not very strongly (3)
  - Don't know/ Prefer not to answer (4)

### `pes21_pidtradstrong`
- Label: À quel point vous sentez-vous proche
- Question: du ${e://Field/pid_party_fr}${pes21_pidtrad/ChoiceTextEntryValue/7}? Quebec
- Options:
  - Très fortement (1)
  - Fortement (2)
  - Pas très fortement (3)
  - Je ne sais pas/Préfère ne pas répondre (4)

### `pes21_langQC`
- Label: In your opinion, is the French language threatened in Quebec?
- Options:
  - Yes (1)
  - No (2)
  - Don’t know/ Prefer not to answer (3)

### `pes21_langQC`
- Label: Selon vous, la langue française est-elle menacée au Québec?
- Options:
  - oui (1)
  - Non (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_cultureQC`
- Label: In your opinion, is the French culture threatened in Quebec?
- Options:
  - Yes (1)
  - No (2)
  - Don’t know/ Prefer not to answer (3)

### `pes21_cultureQC`
- Label: Selon vous, la culture française est-elle menacée au Québec?
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_qclang`
- Label: If Quebec separates from Canada, do you think that the situation of the
- Question: French language in Quebec will...
- Options:
  - Get better (1)
  - Get worse (2)
  - Stay about the same as now (3)
  - Don't know/ Prefer not to answer (4)

### `pes21_qclang`
- Label: Si le Québec se sépare du Canada, pensez-vous que la situation de la
- Question: langue française au Québec...
- Options:
  - Va s'améliorer (1)
  - Va s'aggraver (2)
  - Restera à peu près la même (3)
  - Je ne sais pas/ Préfère ne pas répondre (4)

### `pes21_qcsol`
- Label: If Quebec separates from Canada, do you think your standard of living
- Question: will...
- Options:
  - Get better (1)
  - Get worse (2)
  - Stay about the same as now (3)
  - Don't know/ Prefer not to answer (4)

### `pes21_qcsol`
- Label: Si le Québec se sépare du Canada, croyez-vous que votre niveau de
- Question: vie... Newer lifestyles
- Options:
  - Va s'améliorer (1)
  - Va s'aggraver (2)
  - Restera à peu près le même (3)
  - Je ne sais pas/ Préfère ne pas répondre (4)

### `pes21_newerlife`
- Label: Newer lifestyles are contributing to the breakdown of our society.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_newerlife`
- Label: Les nouveaux modes de vie contribuent à la dégradation de notre
- Question: société. Cognition
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_cognition`
- Label: I like to have responsibility for handling situations that require a lot of
- Question: thinking.
- Options:
  - Strongly disagree (1)
  - Somewhat disagree (2)
  - Neither agree nor disagree (3)
  - Somewhat agree (4)
  - Strongly agree (5)
  - Don't know/ Prefer not to answer (6)

### `pes21_cognition`
- Label: J'aime avoir la responsabilité de gérer des situations qui nécessitent
- Question: beaucoup de réflexion. Femininity and masculinity
- Options:
  - Fortement en désaccord (1)
  - Plutôt en désaccord (2)
  - Ni en accord, ni en désaccord (3)
  - Plutôt en accord (4)
  - Fortement en accord (5)
  - Je ne sais pas/Préfère ne pas répondre (6)

### `pes21_feminine_1`
- Label: How much do you identity as feminine on a scale from 0 to 100,
- Question: where 0 means not at all feminine and 100 means very feminine. Not feminine at all Very feminine 0 10 20 30 40 50 60 70 80 90 100

### `pes21_feminine_1`
- Label: À quel point vous estimez-vous comme féminine sur une échelle de
- Question: 0 à 100, où 0 signifie que vous n'êtes pas du tout féminine et 100 que vous êtes très féminine. Si vous ne savez pas, ou préférez ne pas répondre, veuillez appuyer sur → Pas du tout féminine Très féminine

### `pes21_masculine_1`
- Label: How much do you identity as masculine on a scale from 0 to 100,
- Question: where 0 means not at all masculine and 100 means very masculine. Not masculine at all Very masculine 0 10 20 30 40 50 60 70 80 90 100

### `pes21_masculine_1`
- Label: À quel point vous estimez-vous comme masculin sur une échelle
- Question: de 0 à 100, où 0 signifie que vous n'êtes pas du tout masculin et 100 que vous êtes très masculin. Si vous ne savez pas, ou préférez ne pas répondre, veuillez appuyer sur → Pas du tout masculin Très masculin

### `pes21_big5`
- Label: We’re interested in how you see yourself. Please indicate how well the
- Question: following pair of words describes you, even if one word describes you better than the other. On the left, 1 means those words describe you extremely poorly; on the right, 7 means those words describe you extremely well. Please use any number. Describes you Describes you

### `pes21_big5`
- Label: Nous souhaiterions en apprendre plus sur la façon dont vous vous
- Question: percevez. Indiquez si ces mots vous décrivent bien ou non, même si un des mots vous décrit mieux que l'autre. Sur la gauche, 1 signifie que ces mots vous décrivent très mal; sur la droite, 7 signifie que ces mots vous décrivent très bien. Utiliser n'importe quel nombre.

### `pes21_health`
- Label: Compared to other people your age, how would you describe your
- Question: health?
- Options:
  - Excellent (1)
  - Very good (2)
  - Fair (3)
  - Poor (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_health`
- Label: Comment décririez-vous votre état de santé par rapport aux autres
- Question: personnes de votre âge?
- Options:
  - Excellent (1)
  - Très bien (2)
  - Moyen (3)
  - Pauvre (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_phealth`
- Label: Compared to other people your age, how would you describe your
- Question: physical health?
- Options:
  - Excellent (1)
  - Very good (2)
  - Fair (3)
  - Poor (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_phealth`
- Label: Comment décririez-vous votre état de santé physique par rapport aux
- Question: autres personnes de votre âge?
- Options:
  - Excellent (1)
  - Très bien (2)
  - Moyenne (3)
  - Pauvre (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_mhealth`
- Label: Compared to other people your age, how would you describe your
- Question: mental health?
- Options:
  - Excellent (1)
  - Very good (2)
  - Fair (3)
  - Poor (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_mhealth`
- Label: Comment décririez-vous votre état de santé mentale par rapport aux
- Question: autres personnes de votre âge? The variable FL_128_DO_Health_follow_upphysic gives the display order for pes21_phealth, and the variable FL_128_DO_Health_follow_upmental gives the display order for pes21_mhealth.
- Options:
  - Excellent (1)
  - Très bien (2)
  - Moyenne (3)
  - Pauvre (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_service_freq`
- Label: How often do you attend religious services, excluding special
- Question: occasions such as weddings and funerals?
- Options:
  - Never (1)
  - Once a year (2)
  - Two to eleven times a year (3)
  - Once a month (4)
  - Two or three times a month (5)
  - Once a week or more (6)
  - Don't know / Prefer not to answer (7)

### `pes21_service_freq`
- Label: À quelle fréquence assistez-vous à des cérémonies religieuses,
- Question: en excluant les occasions spéciales telles que les mariages ou les funérailles?
- Options:
  - Jamais (1)
  - Une fois par an (2)
  - Deux à onze fois par an (3)
  - Une fois par mois (4)
  - Deux ou trois fois par mois (5)
  - Une fois par semaine ou plus (6)
  - Je ne sais pas/ Préfère ne pas répondre (7)

### `pes21_parents_born`
- Label: Was either or both of your parents born outside of Canada?
- Options:
  - Yes (1)
  - No (2)
  - Don't know/ Prefer not to answer (3)

### `pes21_parents_born`
- Label: Est-ce qu’au moins l'un(e) de vos parents est né(e) à l'extérieur
- Question: du Canada?
- Options:
  - Oui (1)
  - Non (2)
  - Je ne sais pas/ Préfère ne pas répondre (3)

### `pes21_rural_urban`
- Label: Do you live in…
- Options:
  - A rural area or village (less than1000 people) (1)
  - A small town (more than 1000 people but less than 15K) (2)
  - A middle-sized town (15K-50K people) not attached to a city (3)
  - A suburb of a large town or city (4)
  - A large town or city (more than 50K people) (5)
  - Don't know / Prefer not to answer (6)

### `pes21_rural_urban`
- Label: Vivez-vous...
- Options:
  - En milieu rural ou dans un village (moins de 1000 personnes) (1)
  - Dans une petite ville (plus que 1000 mais moins que 15,000 personnes) (2)
  - Dans une ville de taille moyenne (15k-50k personnes) qui n'est pas adjacente à une grande fille (3)
  - Dans une banlieue d'une grande ville (4)
  - Dans une grande ville (plus que 50k personnes) (5)
  - Je ne sais pas/ Préfère ne pas répondre (6)

### `pes21_lived`
- Label: For how many years have you lived in your current city or community?
- Options:
  - Less than 1 year (1)
  - 1-3 years (2)
  - 3-10 years (3)
  - More than 10 years (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_lived`
- Label: Depuis combien d'années vivez-vous dans votre ville ou communauté
- Question: actuelle?
- Options:
  - Moins d'un an (1)
  - 1-3 ans (2)
  - 3-10 ans (3)
  - Plus de 10 ans (4)
  - Je ne sais pas / Préfère ne pas répondre (5)

### `pes21_follow_pol`
- Label: And how closely do you follow politics on TV, radio, newspapers, or
- Question: the Internet?
- Options:
  - Very closely (1)
  - Fairly closely (2)
  - Not very closely (3)
  - Not at all (4)
  - Don't know/ Prefer not to answer (5)

### `pes21_follow_pol`
- Label: Et à quel point suivez-vous la politique à la télévision, à la radio,
- Question: dans les journaux ou sur l'Internet?
- Options:
  - De très proche (1)
  - D’assez proche (2)
  - Pas de très près (3)
  - Pas du tout de près (4)
  - Je ne sais pas/ Préfère ne pas répondre (5)

### `pes21_lang`
- Label: Which language do you usually speak at home?
- Options:
  - English (1)
  - French (2)
  - Aboriginal language (please specify) (3): pes21_lang_3_TEXT
  - Arabic (4)
  - Chinese, Cantonese, Mandarin (5)
  - Filipino / Tagalog (6)
  - German (7)
  - Indian, Hindi, Gujarati (8)
  - Italian (9)
  - Korean (10)
  - Pakistani, Punjabi, Urdu (11)
  - Persian, Farsi (12)
  - Russian (13)
  - Spanish (14)
  - Tamil (15)
  - Vietnamese (16)
  - Other (please specify) (17): pes21_lang_17_TEXT
  - Don't know/ Prefer not to answer (18)

### `pes21_lang`
- Label: Quelle est la toute première langue que vous avez apprise et que vous
- Question: comprenez encore?
- Options:
  - Anglais (1)
  - Français (2)
  - Langue autochtone (veuillez préciser) (3): pes21_lang_3_TEXT
  - Arabe (4)
  - Chinois, cantonais, mandarin (5)
  - Philippin / tagalog (6)
  - Allemand (7)
  - Indien, Hindi, Gujarati (8)
  - Italien (9)
  - Coréen (10)
  - Pakistanais, Pendjabi, Ourdou (11)
  - Persan, farsi (12)
  - Russe (13)
  - Espagnol (14)
  - Tamil (15)
  - Vietnamien (16)
  - Autre (Veuillez spécifier) (17): pes21_lang_17_TEXT
  - Je ne sais pas / Préfère ne pas répondre (18)

### `pes21_occ_text`
- Label: What is your main occupation? If you are retired, please enter your
- Question: former occupation. ________________________________________________________________

### `pes21_occ_text`
- Label: Quelle est votre occupation principale? Si vous êtes à la retraite,
- Question: veuillez, s’il vous plaît, indiquer votre occupation précédente. Si vous ne savez pas, ou préférez ne pas répondre, veuillez appuyer sur → ________________________________________________________________ Display This Question:

### `pes21_occ_cat`
- Label: Which of the following broad categories best describes your
- Question: occupation? If you do not work currently, select the category of your most recent job.
- Options:
  - Manager (1)
  - Professional (2)
  - Technician or associate professional (3)
  - Clerical support worker (4)
  - Service or sales worker (5)
  - Skilled agricultural, forestry or fishery worker (6)
  - Craft or related trades worker (7)
  - Plant operator, machine operator, or assembler (8)
  - Cleaner, laborer, or assistant (9)
  - Armed forces (10)
  - Other (please specify) (11): pes21_occ_cat_11_TEXT
  - Don't Know/ Prefer not to answer (12)

### `pes21_occ_cat`
- Label: Laquelle des grandes catégories suivantes décrit le mieux votre
- Question: métier? Si vous ne travaillez pas actuellement, sélectionnez la catégorie de votre dernier emploi. (6) PES Data Quality
- Options:
  - Directeurs, cadres de direction et gérants (1)
  - Professions intellectuelles et scientifiques (2)
  - Professions intermédiaires (3)
  - Employés de type administratif (4)
  - Personnel des services directs aux particuliers, commerçants et vendeurs (5)
  - Agriculteurs et ouvriers qualifiés de l’agriculture, de la sylviculture et de la pêche
  - Métiers qualifiés de l’industrie et de l’artisanat (7)
  - Conducteurs d’installations et de machines, et ouvriers de l’assemblage (8)
  - Professions élémentaires (9)
  - Professions militaires (10)
  - Autre (veuillez spécifier) (11): pes21_occ_cat_11_TEXT
  - Je ne sais pas / Préfère ne pas répondre (12)
  - • Valid complete (0)
  - • PES speeder (1)
  - • Duplicate PID (2)
  - • Inattentive (4)
