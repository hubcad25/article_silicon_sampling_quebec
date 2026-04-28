V1.0, 7 mai 2025

1 – Introduction

2 – Définitions

3 – Conditions d’utilisation spécifiques à Narval

4 – Niveau de service Narval

5 – Dispositions finales
1 – Introduction

Ce document consiste en un accord de niveau de service (« ANS ») régissant l'utilisation de la plateforme de calcul de haute performance « Narval ». Conformément aux conditions d'utilisation de CQ, cet ANS comprend des conditions d’utilisation supplémentaires spécifiques à Narval et définit le niveau de service spécifique qui Vous est offert. Cet ANS est conclu entre Calcul Québec (« CQ ») et Vous. Cet ANS entre en vigueur lorsque Vous acceptez ces conditions ou lorsque Vous utilisez pour la première fois l'un des services de Narval.
2 – Définitions

Les définitions relatives à la sécurité de l'information sont disponibles dans le glossaire de la sécurité de l'information de Calcul Québec1.
3 – Conditions d’utilisation spécifiques à Narval

Cet ANS est une extension des conditions d'utilisation de CQ et, en tant que tel, Vous devez Vous conformer aux termes et conditions définis à la fois dans cet ANS et dans les conditions d'utilisation de CQ. Les services Narval Vous sont proposés pour fournir des infrastructures de calcul de haute performance. Cette plateforme et les services qu’elle supporte ne sont pas adaptés au stockage ou au traitement des Informations Sensibles (par exemple les Renseignements personnels et les Renseignements personnels sur la santé). Il est de votre responsabilité de valider si l’utilisation des services de Narval est compatible avec la sensibilité de Vos données.

Les services de Narval Vous sont proposés en tant que PaaS. Ils sont conçus pour Vous fournir un ensemble de ressources partagé par plusieurs Utilisateurs de CQ. Narval Vous permet, par l’entremise d’interfaces prévues à cet effet, d’utiliser les ressources, déployer Vos logiciels et de gérer Vos données.

Les ressources de Narval étant mutualisées, Vos données et informations peuvent être exposées, volontairement ou involontairement, à d'autres utilisateurs de CQ. Vous êtes entièrement responsable d’évaluer les risques que représente cet environnement partagé et de gérer la sécurité et l’accès à Vos données ou informations pour les services de Narval.
4 – Niveau de service Narval

4.1 Spécificités du service

Les services Narval comprennent divers composantes et fonctionnalités qui vous permettent de :

      ● Vous connecter interactivement à votre compte;

      ● Installer des logiciels depuis votre session interactive;

      ● Gérer vos données depuis votre session interactive;

      ● Soumettre des tâches à l'environnement de calcul;

      ● Transférer vos données;

      ● Automatiser certaines tâches.

Les spécificités des services Narval sont disponibles dans la documentation de Narval2. L'utilisation des ressources de calcul de Narval est basée sur une priorité établie en fonction d’une cible propre à chaque projet et de l’utilisation récente des ressources dans ce projet. L’utilisation des ressources de calcul se fait par l’entremise d’un système de soumission de tâches. Il est obligatoire d’utiliser ce système pour utiliser les ressources de calcul de Narval.

Les serveurs de calcul de Narval sont déployés sur une infrastructure qui n'offre pas un niveau élevé de disponibilité et, en tant que tel, n'offre pas de résilience en cas de panne de courant. Un serveur de calcul peut être perdu à tout moment et son utilisation doit être planifiée en conséquence.

Plusieurs composantes de stockage sont offertes avec Narval. Ces espaces de stockage limitent l’espace et le nombre de fichiers avec des quotas. Les composantes HOME et PROJECT ont des copies de sauvegarde tandis que le SCRATCH n’en a pas. Le SCRATCH a également un mécanisme de purge des données qui efface périodiquement les fichiers inutilisés depuis un certain temps. Il est de Votre responsabilité de choisir la bonne composante de stockage et de ne pas conserver à long terme de données sur le SCRATCH, car celles-ci seront perdues.

Certains logiciels couramment utilisés sont disponibles pour l’ensemble des personnes utilisatrices. L’ajout de logiciel peut être fait par Vous, dans Votre compte, ou Vous pouvez demander l’installation via une demande d’assistance.

Les Systèmes d'Information de CQ utilisés pour stocker ou traiter Vos informations sur Narval sont situés uniquement dans la province de Québec.

4.2 Surveillance

Calcul Québec surveillera le fonctionnement et la sécurité des services qu'elle Vous offre. En conséquence, CQ recueillera et traitera divers ensembles de données relatifs à Votre utilisation des services CQ. Ces données sont collectées de manière sécuritaire et sont les suivantes :
Données 	Utilisation
Utilisation des ressources (p.ex. CPU, mémoire, stockage, réseau, etc.) 	Détecter les anomalies, éviter la saturation des ressources et analyser et rapporter l’utilisation, éviter le gaspillage des ressources
Inspection des paquets réseau entrants/sortants (p.ex. origine/destination, protocole, contenu du paquet, etc.) 	Détecter les contenus malveillants et les intrusions, investiguer les incidents de sécurité et monitorer la disponibilité des liens réseaux
Informations relatives à la soumission de tâches (p.ex. paramètre de soumission, script d'exécution, etc) 	Statistiques d’utilisation et activité d’aide aux utilisateurs
Journalisation des requêtes à l'interface utilisateur du service (suivi des activités se déroulant dans une session) 	Créer l’historique des activités et requêtes faites à un système
Surveillance des infrastructures et systèmes de CQ (p.ex. information sur les processus, signature de fichiers, connexion réseau, etc.) 	Détecter les activités anormales et les intrusions, investiguer les incidents de sécurité

4.3 Maintenance

Lorsque CQ planifie une période de maintenance, CQ vous enverra une confirmation de l'événement au minimum deux semaines avant la maintenance. Cette notification doit indiquer la durée prévue de l'interruption ainsi qu’une description sommaire des changements pouvant fortement affecter votre utilisation du service. CQ utilisera l’adresse électronique associée à Votre compte pour la communication ou les notifications. Il est de Votre responsabilité de Vous assurer que cette adresse électronique est valide et à jour.

4.4 Perturbations des services

CQ prend des mesures raisonnables pour éviter toute interruption imprévue des services Narval, mais ne peut pas garantir leur disponibilité ni s'engager à respecter des paramètres de disponibilité prédéfinis. CQ s'efforce de limiter les interruptions de service de Narval lors des opérations de maintenance ou d'autres types d'interventions planifiées. Les interruptions non planifiées occasionnant de longue interruption de service seront communiquées à tous les Utilisateurs de Narval en temps opportun, y compris le temps estimé pour la remise en service.

CQ peut détecter des événements de sécurité ou des anomalies de système sur Narval. Lors d'une telle découverte, et dans la situation où l'événement peut entraîner une large violation de la confidentialité, une perte significative d'informations ou des dommages importants à l'équipement, CQ se réserve le droit, sans aucun préavis, de procéder à une intervention d'urgence qui peut inclure une interruption de service. Dans le cas d'une telle intervention d'urgence, CQ limitera l'étendue et la durée de l'interruption du service et mettra à jour la page de statut3 du service.

4.5 Sauvegardes

CQ sauvegardera son environnement, conformément au Cadre de sécurité de l'information de CQ, pour s'assurer que les services Narval peuvent être ramenés à un état normal aussi rapidement que possible. Les données d’utilisateurs présentent sur les composantes de stockage HOME et PROJET sont sauvegardées généralement une fois par jour. CQ ne garantit pas que ce mécanisme de sauvegarde protège efficacement contre la corruption des données et, par conséquent, si Votre utilisation de Narval nécessite un niveau élevé d'intégrité des données, il Vous incombe de mettre en œuvre les mesures de protection appropriées (par exemple, sauvegardes externe, application ou architecture assurant la résilience des données, etc.)

En outre, Narval ne garantit pas que les données stockées (data at rest) sont chiffrées. Dans le cas où celles-ci doivent être chiffrées, Vous êtes seul responsable de veiller à ce que Votre utilisation du service Narval mette en œuvre le niveau de chiffrement requis.

4.6 Accès aux ressources

Les modalités d’accès aux ressources de Narval sont disponibles dans la documentation4 du service. Les demandes d’allocation pour le service Narval doivent être faites par l’entremise du portail5 de l’Alliance de recherche numérique du Canada.

4.7 Formation et soutien

Les Membres de l'équipe CQ seront disponibles pour Vous fournir des conseils et une assistance de base dans l'utilisation de Narval. Cette assistance sera fournie par courriel6. L'assistance sera disponible de 8 heures à 17 heures, du lundi au vendredi. Cela exclut les jours fériés du Québec. Les réponses aux demandes d'assistance sont en fonction de l’ordre d'arrivée et les délais de réponse sont fonction de l’achalandage et de la disponibilité du personnel.

Aucune formation spécifique au service Narval n’est disponible. Calcul Québec met à Votre disposition plusieurs autres formations7 pertinente à l’utilisation de ce service.
5 – Dispositions finales

CQ peut modifier les termes et conditions de cet ANS à tout moment. Si l'une de ces modifications altère matériellement Vos droits ou Votre utilisation des services Narval, CQ fera des efforts raisonnables pour Vous contacter, notamment en envoyant une notification à l'adresse ou aux adresses électroniques associées à Votre compte. Dans certains cas, il pourra Vous être demandé d'indiquer Votre consentement aux conditions révisées afin de continuer à accéder aux services Narval. Sauf indication contraire, toute modification de cet ANS prendra effet à la date du prochain renouvellement de Votre compte ou lorsque vous Vous connecterez à nouveau à Votre compte. Si Vous n'acceptez pas les conditions révisées, Votre seul et unique recours sera de cesser Votre utilisation de Narval.

    https://www.calculquebec.ca/glossaire-de-la-securite-de-information-CQ ↩

    https://docs.alliancecan.ca/wiki/Narval/fr ↩

    https://status.alliancecan.ca/ ↩

    https://docs.alliancecan.ca/wiki/Narval/fr ↩

    https://ccdb.alliancecan.ca ↩

    support@calculquebec.ca ↩

    https://www.calculquebec.ca/services-aux-chercheurs/formation ↩


ersion 1.0, 19 septembre 2023

1 – Aperçu

2 – Conditions générales

3 – Utilisation des services de CQ

3.1 Accès aux services de CQ

3.2 Utilisation acceptable

4 – Confidentialité et protection des données

5 – Services de CQ hors production (bêta)

6 – Sécurité

7 – Résiliation

8 – Exclusion de responsabilité - Aucune garantie

9 – Limitation de responsabilité

10 – Plaintes ou demandes d'information
1 – Aperçu

Ce document contient les conditions d'utilisation de Calcul Québec (« CQ ») régissant l'utilisation de tous les services de CQ. En plus des conditions d'utilisation de CQ, des accords de niveau de service (« ANS ») contenant des conditions spécifiques aux services que Calcul Québec fournit aux utilisateurs et utilisatrices de CQ devront également être acceptés. Les conditions d'utilisation de CQ, ainsi que tous les ANS applicables, constitueront conjointement l’« Entente ».
2 – Conditions générales

Cette Entente est conclue entre CQ et vous en tant qu'Utilisateur ou Utilisatrice de CQ. Par la présente, vous attestez à CQ que vous disposez de la capacité ou de l'autorité légale nécessaire pour exécuter la présente Entente et que ladite capacité ou autorité légale n'a pas été révoquée, limitée ou modifiée de quelque manière que ce soit. Les termes « Vous », « Votre » ou tout terme connexe en majuscule dans les présentes feront référence à Vous en tant qu'individu. Si Vous ne disposez pas d'une telle capacité ou autorité légale, ou si Vous n'acceptez pas tous les termes et conditions de la présente Entente, Vous ne devez pas accepter la présente Entente et n'êtes donc pas autorisé à utiliser les services de CQ.

Lors de la création ou du renouvellement de Votre compte, vous serez invité à déclarer toute Affiliation existante applicable. Une Affiliation est une organisation, une entité juridique ou un individu vous permettant d'utiliser les services de CQ pour des utilisations ou des activités autorisées. Dans le cas où l'Affiliation a un accord de service avec CQ, CQ s'engage par les présentes à respecter cet accord, y compris les dispositions relatives au transfert ou au partage de votre responsabilité à l'Affiliation. Vous avez l’entière responsabilité de vous informer et de respecter toutes les exigences, conditions, limitations ou modalités supplémentaires relatives à un tel accord de service.

La présente Entente doit être interprétée et est régie par les lois de la province de Québec, Canada, et les lois fédérales du Canada qui y sont applicables, à l'exclusion de toute disposition conflictuelle relative à ces lois. Toute réclamation ou procédure intentée par une partie contre toute autre partie en relation avec la présente Entente sera traitée sous la juridiction des tribunaux du Québec.

Vous ne pouvez pas céder ou transférer la présente Entente ou Vos droits en vertu des présentes, en totalité ou en partie, par effet de la loi ou autrement, sans le consentement écrit préalable de CQ. Aucune renonciation à une disposition de la présente Entente, y compris une renonciation à une violation de la présente Entente, ne constitue une renonciation à toute autre disposition ou violation de la présente Entente, sauf disposition expresse contraire. Aucune renonciation ne sera contraignante à moins d’être par écrit et signée par les deux parties. Dans le cas où une partie de la présente Entente est jugée invalide ou inapplicable, la partie inapplicable prendra effet dans toute la mesure du possible et les parties restantes resteront pleinement en vigueur.

CQ se réserve le droit d'apporter des améliorations, des développements ou des modifications à la présente Entente à tout moment. Si une révision modifie sensiblement Vos droits, CQ déploiera des efforts raisonnables pour Vous contacter, notamment en envoyant une notification à la ou aux adresses courriel associées à Votre compte. Dans certaines situations, vous devrez peut-être indiquer Votre consentement aux conditions révisées afin de continuer à accéder aux services de CQ. Sauf indication contraire, toute modification de la présente Entente prendra effet à la date à laquelle il y aura un accès ou utilisation de Votre compte. Si Vous n'êtes pas d'accord avec les conditions révisées, Votre seul et unique recours sera de ne pas utiliser les services de CQ et de fermer Votre compte aux services de CQ.

Toutes les communications et avis effectués ou donnés en vertu de la présente Entente, des politiques et des services de CQ doivent être en français ou en anglais.
3 – Utilisation des services de CQ

3.1 Accès aux services de CQ

Sous réserve de Votre respect de la présente Entente, CQ vous accorde par la présente un droit limité, révocable, non exclusif, non transférable et ne pouvant faire l'objet d'une sous-licence, d'accès et d'utilisation des services de CQ.

Si votre accès aux services CQ vous accorde le rôle de Parrain, CQ vous permettra de fournir l'accès et l'utilisation des services de CQ à des Utilisateurs ou Utilisatrices parrainés. Un Utilisateur parrainé ou une Utilisatrice parrainée désigne toute personne ou entité autorisée ou mandatée pour accéder et utiliser les services de CQ sous Votre responsabilité. Tout Utilisateur parrainé peut accéder et utiliser les services de CQ à condition qu'il ou elle accepte d'être lié par les termes de la présente Entente et d'utiliser les ressources conformément aux objectifs définis par le Parrain. Le Parrain est responsable d'accorder et de révoquer l'accès de ses Utilisateurs parrainés et d'identifier et de communiquer à ses Utilisateurs parrainés les usages autorisés associés à l'utilisation des services CQ.

CQ n'est pas responsable de l'accès non autorisé à Votre compte. Vous êtes responsable de la sécurité et de la confidentialité de Vos informations, y compris de la collecte et de la conservation des informations relatives au consentement et de la conservation de Vos informations.

Les comptes CQ sont privés et personnels et Vous ne devez jamais accéder ou tenter d'accéder à des services de CQ autres que les ressources ou données auxquelles Vous avez explicitement accès. Les identifiants de Votre compte sont confidentiels et doivent rester secrets et sécurisés à tout moment, conformément aux politiques de CQ. Vous ne devez autoriser aucune autre personne à utiliser Vos informations d'identification pour accéder aux services de CQ. Si du personnel supplémentaire a besoin d'accéder aux services de CQ, chaque personne doit demander un compte et obtenir ses propres informations d'identification. Vous êtes seul responsable de toutes les activités effectuées par et dans Votre compte.

Si vous avez connaissance d'une brèche ou d'une violation de Votre compte, Vous devez prendre immédiatement toutes les mesures nécessaires pour suspendre l'accès à Votre compte et à Votre contenu et remédier à la situation. Si l'un de ces éléments entraîne un risque de confidentialité ou de sécurité pour CQ, Vous en informerez immédiatement CQ1.

CQ se réserve le droit de modifier, d'interrompre ou de mettre fin à ses services à tout moment. Ces changements incluent, sans s'y limiter, la modification ou l'arrêt de certaines fonctionnalités des services de CQ.

3.2 Utilisation acceptable

En tant qu'Utilisateur ou Utilisatrice de CQ, Vous bénéficiez d'un ensemble diversifié de services offerts par Calcul Québec. Dans le contexte de ces services et dans le cadre de la présente Entente, Vous acceptez, sauf autorisation expresse écrite de CQ, de ne pas :

I. Participer à toute activité qui contrevient aux lois canadiennes ou québécoises;

II. Entreprendre toute action susceptible de porter atteinte aux droits de propriété intellectuelle de quelqu'un d'autre;

III. Utiliser les ressources de CQ dans une situation pour laquelle la défaillance ou la faute de celle-ci pourrait entraîner la mort ou des blessures graves de toute personne ou animal, ou de graves dommages physiques ou environnementaux;

IV. Introduire dans l'environnement de CQ des logiciels malveillants susceptibles d'endommager, d'interférer ou de capturer tout système, programme ou donnée en dehors des ressources qui Vous sont assignées ou auxquelles Vous avez explicitement accès;

V. Tenter de sonder, analyser, pénétrer ou tester les vulnérabilités des services de CQ ou de ses systèmes;

VI. Tenter d'accéder ou d'intercepter des données qui ne Vous sont pas destinées ou qui ne sont pas destinées aux ressources qui Vous sont attribuées ou auxquelles Vous avez accès;

VII. Utiliser les ressources de CQ d'une manière qui créerait une charge excessive sur les services de CQ;

VIII. Envoyer ou faciliter des communications de masse non sollicitées; IX. Utiliser des applications sans licence valide et appropriée;

X. Utiliser des services ou logiciels qui outrepassent l'authentification ou l'application des politiques de CQ;

XI. Utiliser des logiciels ou des services qui fournissent un accès non protégé aux services de CQ;

XII. Vendre, revendre, louer, céder ou fournir tout accès à un tiers qui n'est pas autrement défini dans la présente Entente.
4 – Confidentialité et protection des données

En prenant part à cette Entente, Vous acceptez de Vous conformer à toutes Vos obligations telles que définies dans la politique de confidentialité et de protection des données de CQ2. Sauf entente écrite entre Vous et CQ à l'effet contraire, Vous conserverez tous les droits, titres et intérêts sur les données que Vous transmettez, traitez ou stockez via l'utilisation des services de CQ. Aux fins de la présente Entente et du reste des politiques de CQ, cela sera appelé Vos données ou Données de l’Utilisateur ou de l’Utilisatrice de CQ.

Vous êtes responsable d’obtenir tous les droits ou consentements nécessaires avant d’utiliser les services de CQ. Vous serez seul responsable de l’utilisation, de la divulgation, du stockage et de la transmission de Vos données. CQ n'assume aucune responsabilité quant à l'utilisation, la divulgation, le stockage et la transmission que vous faites de Vos données.

Dans le cas où Vous avez déclaré une ou des affiliations à CQ, Vous acceptez par la présente que CQ partage des informations pertinentes, y compris des Renseignements personnels, sur Votre compte et Votre utilisation des services de CQ, ainsi que toute autre donnée décrite dans un accord de service applicable, avec la ou les affiliations. Dans le cas où Votre compte est parrainé, Vous acceptez également que CQ partage à Votre parrain les informations pertinentes relatives à Votre utilisation des services de CQ.

CQ n'accédera ni n'utilisera Vos données, sauf dans les conditions énoncées dans la politique de confidentialité et de protection des données de CQ ou selon les modalités définies dans le cadre d'un ANS.
5 – Services de CQ hors production (bêta)

Les services de CQ hors production (par exemple, les systèmes en mode bêta ou systèmes de test) font référence aux services proposés par CQ qui, bien qu'ils incluent toutes les fonctionnalités prévues, ne sont pas encore entièrement testés pour leur sécurité et leurs performances optimales. Ces services seront proposés « tels quels » et Vous pouvez donc Vous attendre à des pannes, des maintenances ou des interruptions de service fréquentes. Un service hors production est temporaire et peut être interrompu ou retiré à tout moment, avec ou sans préavis. CQ fera de son mieux pour Vous informer à l'avance de l'interruption et la cessation d’un service bêta afin que Vous puissiez prendre les mesures nécessaires pour sécuriser, sauvegarder ou transférer Vos données. Les données hébergées sur des services hors production peuvent être perdues et CQ n'assume aucune responsabilité quant aux données qui y sont stockées. CQ peut fournir une assistance et un soutien limité pour l'utilisation des services hors production, comme elle l’estime nécessaire. Ces services seront identifiés comme tels et n’auront aucun ANS associé.
6 – Sécurité

CQ met en œuvre des contrôles de sécurité pour protéger Votre compte et Vos données sur ses Systèmes d'information. Cependant, lorsque Vous utilisez les services de CQ, Vous transmettez également des informations ou des données sur des réseaux qui ne sont pas exploités ou gérés par CQ. CQ n'est pas responsable de la sécurité de Vos données ou de Vos informations lors de l'utilisation de ces réseaux tiers et ne garantit pas que Vos informations ne seront pas perdues, altérées ou corrompues sur ces réseaux tiers.

CQ utilise et améliore constamment son Cadre de sécurité de l'information pour faire face aux risques et menaces connus. Cependant, CQ ne peut offrir une garantie absolue que ses mesures de sécurité empêcheront un accès non autorisé à Vos données. Votre conformité aux politiques de CQ est un élément essentiel de la sécurisation des services de CQ.
7 – Résiliation

Une violation de toute disposition de la présente Entente peut entraîner la suspension de Votre compte ou la révocation de Vos privilèges d'accès. CQ se réserve le droit de résilier Votre compte, à tout moment, avec ou sans préavis, si vous manquez à l'une de Vos obligations à plusieurs reprises ou si Vous ne parvenez pas à remédier à Votre manquement en temps opportun.

Vous pouvez fermer Votre compte à tout moment en contactant l'équipe d'assistance de CQ par écrit3.

Une fois l’Entente résiliée par Vous ou par CQ, tous les droits et accès aux Services de CQ prendront fin, sauf indication contraire expresse dans la présente Entente.
8 – Exclusion de responsabilité - Aucune garantie

BIEN QUE CALCUL QUÉBEC S'ENGAGE À MAINTENIR L'INTÉGRITÉ ET LA SÉCURITÉ DE SES SERVICES DANS LA PLUS LARGE MESURE POSSIBLE, LES SERVICES DE CALCUL QUÉBEC FOURNIS DANS CETTE ENTENTE PEUVENT CONTENIR DES BOGUES, DES ERREURS, DES PROBLÈMES DE PERFORMANCE, DES DÉFAUTS OU DES COMPOSANTS NUISIBLES. EN CONSÉQUENCE, CALCUL QUÉBEC VOUS FOURNIT SES SERVICES « TELS QUELS ». CELA SIGNIFIE QUE, DANS LA MESURE PERMISE PAR LA LOI, ET EN ABSENCE DE FAUTE INTENTIONNELLE OU GRAVE DE CALCUL QUÉBEC, CALCUL QUÉBEC ET SES AFFILIÉS NE FONT AUCUNE DÉCLARATION OU N’OFFRE DE GARANTIE D'AUCUNE SORTE, EXPRESSE, IMPLICITE, LÉGALE OU AUTRE CONCERNANT LES SERVICES DE CALCUL QUÉBEC, INCLUANT TOUTE GARANTIE DE QUALITÉ MARCHANDE, D'ADAPTATION À UN USAGE PARTICULIER, DE NON-VIOLATION ET DE TITRE. CALCUL QUÉBEC ET SES AFFILIÉS NE FONT AUCUNE DÉCLARATION NI GARANTIE QUE LES SERVICES DE CALCUL QUÉBEC SERONT DISPONIBLES, ININTERROMPUS, SANS ERREUR OU EXEMPT D'ÉLÉMENTS NUISIBLES OU QUE TOUT CONTENU, Y COMPRIS VOTRE CONTENU, SERA SÉCURISÉ OU NON PERDU OU ENDOMMAGÉ. AINSI, VOTRE UTILISATION DES SERVICES DE CALCUL QUÉBEC EST À VOTRE PROPRE DISCRÉTION ET À VOS PROPRES RISQUES. DE PLUS, IL EST EXPRESSÉMENT ENTENDU QUE LES SERVICES DE CALCUL QUÉBEC SONT MIS À VOTRE DISPOSITION SANS QUE CALCUL QUÉBEC N'AIT AUCUNE OBLIGATION DE SURVEILLER, DE CONTRÔLER OU DE VÉRIFIER LE CONTENU OU LES DONNÉES DES UTILISATEURS.

LES EXCLUSIONS DE NON-RESPONSABILITÉ MENTIONNÉES DANS CETTE ENTENTE NE LIMITENT NI N'AFFECTENT AUCUNE EXCLUSION DE NON-RESPONSABILITÉ DANS UNE ENTENTE AVEC CALCUL QUÉBEC, LES CONDITIONS D'UTILISATION, UN ANS ET DANS TOUTE AUTRE DE SES POLITIQUES.
9 – Limitation de responsabilité

LA RESPONSABILITÉ DE CALCUL QUÉBEC, LE CAS ÉCHÉANT, EST LIMITÉE UNIQUEMENT AUX DOMMAGES DIRECTS CAUSÉS PAR SA FAUTE INTENTIONNELLE OU GRAVE. CALCUL QUÉBEC DÉCLINE TOUTE RESPONSABILITÉ POUR LES DOMMAGES CONSÉCUTIFS, ACCESSOIRES, EXEMPLAIRES OU PUNITIFS, OU POUR TOUTE PERTE DE PROFITS, DE REVENUS, D'AFFAIRES, DE DONNÉES OU D'UTILISATION DE DONNÉES DÉCOULANT DE L'UTILISATION DES SERVICES CALCUL QUÉBEC. NI CALCUL QUÉBEC NI SES AFFILIÉS NE SERONT RESPONSABLES DE TOUT DOMMAGE RÉSULTANT DE DÉFAUTS DE TOUT LOGICIEL OU MATÉRIEL TIERS LIÉ À L'INFRASTRUCTURE ET AUX SERVICES DE CALCUL QUÉBEC, SAUF DANS LA MESURE CAUSÉE PAR LA PROPRE NÉGLIGENCE INTENTIONNELLE OU GRAVE DE CALCUL QUÉBEC.
10 – Plaintes ou demandes d'information

Si Vous avez une demande ou souhaitez déposer une plainte concernant la présente Entente, Votre compte ou les services de CQ qui Vous ont été proposés, Vous devez contacter le support de CQ à support@calculquebec.ca et nous répondrons à Votre demande ou plainte, et si possible, la résoudrons en temps opportun.

    securite@calculquebec.ca ↩

    https://www.calculquebec.ca/politique-de-confidentialité-et-de-protection-des-donnees ↩

    support@calculquebec.ca ↩

