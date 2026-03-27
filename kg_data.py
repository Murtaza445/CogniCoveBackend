"""
Knowledge Graph data definitions for CogniCove.

All data extracted manually from DSM-5 chunks in disorders_chunks.py.
Disorder names and category strings match metadata values exactly.

Three sections:
  A — COMORBIDITY_EDGES: bidirectional comorbidity relationships with weights
  B — DISORDER_CRITERIA: symptom lists, thresholds, duration rules per disorder
  C — EXCLUSION_LINKS + DIFFERENTIAL_PAIRS: exclusion and differential relationships
"""

# ─────────────────────────────── CATEGORIES ───────────────────────────────

CATEGORY_MAP = {
    "Major Depressive Disorder": "Depressive Disorders",
    "Persistent Depressive Disorder": "Depressive Disorders",
    "Substance/Medication-Induced Depressive Disorder": "Depressive Disorders",
    "Disruptive Mood Dysregulation Disorder": "Depressive Disorders",
    "Depressive Disorder Due to Another Medical Condition": "Depressive Disorders",
    "Separation Anxiety Disorder": "Anxiety Disorder",
    "Specific Phobia": "Anxiety Disorder",
    "Selective Mutism": "Anxiety Disorder",
    "Social Anxiety Disorder": "Anxiety Disorder",
    "Panic Disorder": "Anxiety Disorder",
    "Reactive Attachment Disorder": "Trauma- and Stressor-Related Disorders",
    "Disinhibited Social Engagement Disorder": "Trauma- and Stressor-Related Disorders",
    "Posttraumatic Stress Disorder": "Trauma- and Stressor-Related Disorders",
    "Prolonged Grief Disorder": "Trauma- and Stressor-Related Disorders",
    "Acute Stress Disorder": "Trauma- and Stressor-Related Disorders",
    "Obsessive-Compulsive Disorder": "Obsessive-Compulsive and Related Disorders",
    "Body Dysmorphic Disorder": "Obsessive-Compulsive and Related Disorders",
    "Hoarding Disorder": "Obsessive-Compulsive and Related Disorders",
    "Excoriation (Skin-Picking) Disorder": "Obsessive-Compulsive and Related Disorders",
    "Trichotillomania (Hair-Pulling Disorder)": "Obsessive-Compulsive and Related Disorders",
    "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder": "Obsessive-Compulsive and Related Disorders",
    "Bipolar I Disorder": "Bipolar and related Disorder",
    "Bipolar II Disorder": "Bipolar and related Disorder",
    "Cyclothymic Disorder": "Bipolar and related Disorder",
    "Substance/Medication-Induced Bipolar and Related Disorder": "Bipolar and related Disorder",
}

# ──────────────────────── SECTION A: COMORBIDITY EDGES ────────────────────────
# (source_disorder, target_disorder, weight, evidence_note)
# Weights estimated from DSM-5 prevalence data in CHUNK 8 entries.
# Both (A→B) and (B→A) are added programmatically in knowledge_graph.py.

COMORBIDITY_EDGES = [
    # ── MDD comorbidities ──
    ("Major Depressive Disorder", "Panic Disorder", 0.50, "Panic disorder commonly co-occurs with MDD"),
    ("Major Depressive Disorder", "Generalized Anxiety Disorder", 0.50, "High rates of co-occurrence with MDD"),  # GAD is not a separate disorder in our set
    ("Major Depressive Disorder", "Posttraumatic Stress Disorder", 0.45, "Frequently comorbid, especially following trauma"),
    ("Major Depressive Disorder", "Obsessive-Compulsive Disorder", 0.35, "Notable co-occurrence with MDD"),
    ("Major Depressive Disorder", "Social Anxiety Disorder", 0.30, "Social anxiety commonly comorbid"),
    ("Major Depressive Disorder", "Specific Phobia", 0.25, "Specific phobia co-occurs with MDD"),
    ("Major Depressive Disorder", "Persistent Depressive Disorder", 0.40, "PDD may co-occur with MDD (double depression)"),
    ("Major Depressive Disorder", "Bipolar I Disorder", 0.15, "Depressive episodes overlap; differential needed"),
    ("Major Depressive Disorder", "Bipolar II Disorder", 0.15, "Depressive episodes overlap; differential needed"),

    # ── PDD comorbidities ──
    ("Persistent Depressive Disorder", "Major Depressive Disorder", 0.50, "Double depression common"),
    ("Persistent Depressive Disorder", "Separation Anxiety Disorder", 0.30, "Elevated in PDD"),
    ("Persistent Depressive Disorder", "Social Anxiety Disorder", 0.30, "Elevated in PDD"),
    ("Persistent Depressive Disorder", "Specific Phobia", 0.25, "Elevated in PDD"),

    # ── Substance/Med-Induced Depressive ──
    ("Substance/Medication-Induced Depressive Disorder", "Major Depressive Disorder", 0.40, "Higher overall psychiatric comorbidity than independent MDD"),

    # ── DMDD comorbidities ──
    ("Disruptive Mood Dysregulation Disorder", "Major Depressive Disorder", 0.45, "MDD and dysthymia most common comorbidities"),
    ("Disruptive Mood Dysregulation Disorder", "Separation Anxiety Disorder", 0.30, "Anxiety disorders frequently comorbid with DMDD"),

    # ── Depressive Due to Medical Condition ──
    ("Depressive Disorder Due to Another Medical Condition", "Major Depressive Disorder", 0.30, "Depressive features overlap; medical etiology primary"),

    # ── Separation Anxiety ──
    ("Separation Anxiety Disorder", "Specific Phobia", 0.45, "Highly comorbid in children"),
    ("Separation Anxiety Disorder", "Social Anxiety Disorder", 0.35, "Comorbid in adults"),
    ("Separation Anxiety Disorder", "Panic Disorder", 0.35, "Comorbid in adults"),
    ("Separation Anxiety Disorder", "Posttraumatic Stress Disorder", 0.30, "Comorbid in adults"),
    ("Separation Anxiety Disorder", "Obsessive-Compulsive Disorder", 0.25, "Comorbid in adults"),

    # ── Specific Phobia ──
    ("Specific Phobia", "Major Depressive Disorder", 0.35, "Increased risk of developing MDD"),
    ("Specific Phobia", "Bipolar I Disorder", 0.25, "Increased risk of developing bipolar disorders"),
    ("Specific Phobia", "Bipolar II Disorder", 0.25, "Increased risk of developing bipolar disorders"),

    # ── Social Anxiety Disorder ──
    ("Social Anxiety Disorder", "Major Depressive Disorder", 0.50, "High comorbidity; chronic isolation may cause MDD"),
    ("Social Anxiety Disorder", "Specific Phobia", 0.35, "Comorbid anxiety disorders"),
    ("Social Anxiety Disorder", "Separation Anxiety Disorder", 0.30, "Comorbid anxiety disorders"),
    ("Social Anxiety Disorder", "Body Dysmorphic Disorder", 0.25, "Appearance anxiety overlaps"),

    # ── Panic Disorder ──
    ("Panic Disorder", "Major Depressive Disorder", 0.55, "Lifetime comorbidity 10-65%; two-thirds MDD follows panic"),
    ("Panic Disorder", "Bipolar I Disorder", 0.30, "Elevated comorbidity"),
    ("Panic Disorder", "Bipolar II Disorder", 0.30, "Elevated comorbidity"),
    ("Panic Disorder", "Social Anxiety Disorder", 0.35, "Common comorbid anxiety"),
    ("Panic Disorder", "Specific Phobia", 0.35, "Common comorbid anxiety"),
    ("Panic Disorder", "Obsessive-Compulsive Disorder", 0.25, "OCD comorbid with panic"),
    ("Panic Disorder", "Posttraumatic Stress Disorder", 0.30, "PTSD comorbid with panic"),

    # ── Reactive Attachment Disorder ──
    ("Reactive Attachment Disorder", "Disinhibited Social Engagement Disorder", 0.30, "Both arise from neglect; may co-occur"),

    # ── Disinhibited Social Engagement Disorder ──
    ("Disinhibited Social Engagement Disorder", "Reactive Attachment Disorder", 0.30, "Both arise from neglect"),

    # ── PTSD ──
    ("Posttraumatic Stress Disorder", "Major Depressive Disorder", 0.55, "Frequently comorbid depressive disorders"),
    ("Posttraumatic Stress Disorder", "Bipolar I Disorder", 0.25, "Elevated bipolar comorbidity"),
    ("Posttraumatic Stress Disorder", "Bipolar II Disorder", 0.25, "Elevated bipolar comorbidity"),
    ("Posttraumatic Stress Disorder", "Panic Disorder", 0.30, "Comorbid anxiety disorders"),
    ("Posttraumatic Stress Disorder", "Social Anxiety Disorder", 0.30, "Comorbid anxiety disorders"),
    ("Posttraumatic Stress Disorder", "Obsessive-Compulsive Disorder", 0.25, "OCD co-occurrence elevated"),

    # ── OCD ──
    ("Obsessive-Compulsive Disorder", "Major Depressive Disorder", 0.41, "MDD comorbid in 41% of OCD cases"),
    ("Obsessive-Compulsive Disorder", "Panic Disorder", 0.40, "Anxiety disorders comorbid in 76%"),
    ("Obsessive-Compulsive Disorder", "Social Anxiety Disorder", 0.40, "Anxiety disorders comorbid in 76%"),
    ("Obsessive-Compulsive Disorder", "Specific Phobia", 0.35, "Anxiety disorders comorbid in 76%"),
    ("Obsessive-Compulsive Disorder", "Body Dysmorphic Disorder", 0.30, "OC-related disorders at elevated rates"),
    ("Obsessive-Compulsive Disorder", "Trichotillomania (Hair-Pulling Disorder)", 0.25, "OC-related disorders at elevated rates"),
    ("Obsessive-Compulsive Disorder", "Excoriation (Skin-Picking) Disorder", 0.25, "OC-related disorders at elevated rates"),
    ("Obsessive-Compulsive Disorder", "Bipolar I Disorder", 0.20, "Bipolar comorbidity elevated above general population"),

    # ── BDD ──
    ("Body Dysmorphic Disorder", "Major Depressive Disorder", 0.55, "Most common comorbid condition; MDD onset usually follows BDD"),
    ("Body Dysmorphic Disorder", "Social Anxiety Disorder", 0.40, "Commonly comorbid"),
    ("Body Dysmorphic Disorder", "Obsessive-Compulsive Disorder", 0.35, "Frequently comorbid, overlapping features"),

    # ── Hoarding Disorder ──
    ("Hoarding Disorder", "Major Depressive Disorder", 0.45, "MDD comorbid in 30-50%"),
    ("Hoarding Disorder", "Social Anxiety Disorder", 0.30, "Frequently comorbid"),
    ("Hoarding Disorder", "Obsessive-Compulsive Disorder", 0.20, "~20% also have OCD"),

    # ── Excoriation ──
    ("Excoriation (Skin-Picking) Disorder", "Obsessive-Compulsive Disorder", 0.35, "Frequently comorbid"),
    ("Excoriation (Skin-Picking) Disorder", "Trichotillomania (Hair-Pulling Disorder)", 0.40, "Frequently comorbid; both BFRBDs"),
    ("Excoriation (Skin-Picking) Disorder", "Major Depressive Disorder", 0.35, "Commonly comorbid; more prevalent in women"),

    # ── Trichotillomania ──
    ("Trichotillomania (Hair-Pulling Disorder)", "Major Depressive Disorder", 0.40, "Most common comorbid condition"),
    ("Trichotillomania (Hair-Pulling Disorder)", "Excoriation (Skin-Picking) Disorder", 0.40, "Frequently comorbid; both BFRBDs"),

    # ── Bipolar I ──
    ("Bipolar I Disorder", "Panic Disorder", 0.40, "Anxiety disorders most frequently comorbid"),
    ("Bipolar I Disorder", "Social Anxiety Disorder", 0.35, "Anxiety disorders most frequently comorbid"),
    ("Bipolar I Disorder", "Specific Phobia", 0.30, "Anxiety disorders most frequently comorbid"),
    ("Bipolar I Disorder", "Major Depressive Disorder", 0.35, "Depressive episodes characteristic"),
    ("Bipolar I Disorder", "Posttraumatic Stress Disorder", 0.25, "Elevated PTSD comorbidity"),
    ("Bipolar I Disorder", "Obsessive-Compulsive Disorder", 0.20, "OCD comorbidity elevated"),

    # ── Bipolar II ──
    ("Bipolar II Disorder", "Social Anxiety Disorder", 0.38, "Social anxiety comorbid in 38%"),
    ("Bipolar II Disorder", "Specific Phobia", 0.36, "Specific phobia comorbid in 36%"),
    ("Bipolar II Disorder", "Panic Disorder", 0.30, "Anxiety disorders 75% comorbid"),
    ("Bipolar II Disorder", "Major Depressive Disorder", 0.45, "Depressive episodes core feature; MDD comorbid"),
    ("Bipolar II Disorder", "Posttraumatic Stress Disorder", 0.20, "Lower PTSD rates than bipolar I"),

    # ── Cyclothymic Disorder ──
    ("Cyclothymic Disorder", "Bipolar I Disorder", 0.35, "15-50% risk of converting to bipolar I or II"),
    ("Cyclothymic Disorder", "Bipolar II Disorder", 0.35, "15-50% risk of converting to bipolar I or II"),

    # ── Substance/Med-Induced Bipolar ──
    ("Substance/Medication-Induced Bipolar and Related Disorder", "Bipolar I Disorder", 0.20, "May suggest underlying bipolar diathesis"),
]


# ──────────────────────── SECTION B: DISORDER CRITERIA ────────────────────────
# Extracted from CHUNK 2 (symptom_list) and CHUNK 3 (duration) per disorder.
# Each entry lists canonical symptom names, diagnostic threshold, and duration rule.

DISORDER_CRITERIA = {
    "Major Depressive Disorder": {
        "symptoms": [
            "depressed mood",
            "anhedonia",
            "weight or appetite change",
            "sleep disturbance",
            "psychomotor agitation or retardation",
            "fatigue or loss of energy",
            "worthlessness or excessive guilt",
            "concentration difficulty or indecisiveness",
            "suicidal ideation or recurrent thoughts of death",
        ],
        "threshold": 5,
        "must_include_one_of": ["depressed mood", "anhedonia"],
        "duration_days": 14,
        "duration_note": "At least 2 weeks, most of the day, nearly every day",
    },
    "Persistent Depressive Disorder": {
        "symptoms": [
            "depressed mood most of the day, more days than not",
            "poor appetite or overeating",
            "insomnia or hypersomnia",
            "low energy or fatigue",
            "low self-esteem",
            "poor concentration or difficulty making decisions",
            "feelings of hopelessness",
        ],
        "threshold": 2,
        "must_include_one_of": ["depressed mood most of the day, more days than not"],
        "duration_days": 730,
        "duration_note": "At least 2 years in adults; 1 year in children/adolescents",
    },
    "Substance/Medication-Induced Depressive Disorder": {
        "symptoms": [
            "depressed mood",
            "diminished interest or pleasure",
        ],
        "threshold": 1,
        "must_include_one_of": ["depressed mood", "diminished interest or pleasure"],
        "duration_days": None,
        "duration_note": "During or soon after substance intoxication/withdrawal; resolves within ~1 month",
    },
    "Disruptive Mood Dysregulation Disorder": {
        "symptoms": [
            "severe recurrent temper outbursts",
            "temper outbursts inconsistent with developmental level",
            "persistent irritable or angry mood between outbursts",
        ],
        "threshold": 3,
        "must_include_one_of": ["severe recurrent temper outbursts", "persistent irritable or angry mood between outbursts"],
        "duration_days": 365,
        "duration_note": "At least 12 months; temper outbursts 3+ times per week; age 6-18",
    },
    "Depressive Disorder Due to Another Medical Condition": {
        "symptoms": [
            "depressed mood",
            "diminished interest or pleasure",
        ],
        "threshold": 1,
        "must_include_one_of": ["depressed mood", "diminished interest or pleasure"],
        "duration_days": None,
        "duration_note": "Direct physiological consequence of a medical condition",
    },
    "Separation Anxiety Disorder": {
        "symptoms": [
            "distress when anticipating or experiencing separation",
            "worry about losing attachment figures",
            "worry about untoward events causing separation",
            "reluctance to go out due to fear of separation",
            "fear of being alone without attachment figures",
            "reluctance to sleep away from attachment figures",
            "nightmares involving separation themes",
            "physical complaints when separation occurs or is anticipated",
        ],
        "threshold": 3,
        "must_include_one_of": [],
        "duration_days": 120,
        "duration_note": "At least 4 weeks in children; 6 months+ in adults",
    },
    "Specific Phobia": {
        "symptoms": [
            "marked fear or anxiety about a specific object or situation",
            "phobic object or situation almost always provokes immediate fear",
            "phobic object or situation is actively avoided or endured with intense anxiety",
            "fear or anxiety is out of proportion to actual danger",
            "fear, anxiety, or avoidance is persistent",
        ],
        "threshold": 5,
        "must_include_one_of": [],
        "duration_days": 180,
        "duration_note": "At least 6 months",
    },
    "Selective Mutism": {
        "symptoms": [
            "consistent failure to speak in specific social situations despite speaking in others",
            "interferes with educational or occupational achievement or social communication",
        ],
        "threshold": 2,
        "must_include_one_of": ["consistent failure to speak in specific social situations despite speaking in others"],
        "duration_days": 30,
        "duration_note": "At least 1 month (not limited to first month of school)",
    },
    "Social Anxiety Disorder": {
        "symptoms": [
            "marked fear or anxiety about social situations with possible scrutiny",
            "fear of showing anxiety symptoms that will be negatively evaluated",
            "social situations almost always provoke fear or anxiety",
            "social situations are avoided or endured with intense anxiety",
            "fear or anxiety is out of proportion to actual threat",
            "fear, anxiety, or avoidance is persistent",
        ],
        "threshold": 6,
        "must_include_one_of": [],
        "duration_days": 180,
        "duration_note": "At least 6 months",
    },
    "Panic Disorder": {
        "symptoms": [
            "palpitations or accelerated heart rate",
            "sweating",
            "trembling or shaking",
            "shortness of breath or smothering",
            "feelings of choking",
            "chest pain or discomfort",
            "nausea or abdominal distress",
            "dizziness or lightheadedness",
            "chills or heat sensations",
            "paresthesias (numbness or tingling)",
            "derealization or depersonalization",
            "fear of losing control or going crazy",
            "fear of dying",
        ],
        "threshold": 4,
        "must_include_one_of": [],
        "duration_days": None,
        "duration_note": "Recurrent unexpected panic attacks; persistent concern or maladaptive change ≥1 month",
    },
    "Reactive Attachment Disorder": {
        "symptoms": [
            "rarely seeks comfort when distressed",
            "rarely responds to comfort when distressed",
            "minimal social and emotional responsiveness to others",
            "limited positive affect",
            "unexplained irritability, sadness, or fearfulness during non-threatening interactions",
        ],
        "threshold": 2,
        "must_include_one_of": ["rarely seeks comfort when distressed", "rarely responds to comfort when distressed"],
        "duration_days": None,
        "duration_note": "Pattern of insufficient care before age 5; developmental age ≥9 months",
    },
    "Disinhibited Social Engagement Disorder": {
        "symptoms": [
            "reduced or absent reticence approaching unfamiliar adults",
            "overly familiar verbal or physical behavior",
            "diminished or absent checking back with caregiver after venturing away",
            "willingness to go off with unfamiliar adult with minimal or no hesitation",
        ],
        "threshold": 2,
        "must_include_one_of": [],
        "duration_days": None,
        "duration_note": "Pattern of insufficient care before age 5; developmental age ≥9 months",
    },
    "Posttraumatic Stress Disorder": {
        "symptoms": [
            "intrusive distressing memories of the traumatic event",
            "recurrent distressing dreams related to the event",
            "dissociative reactions (flashbacks)",
            "intense psychological distress at trauma cues",
            "marked physiological reactions to trauma cues",
            "avoidance of distressing memories, thoughts, or feelings",
            "avoidance of external reminders of the event",
            "inability to remember important aspects of the event",
            "persistent negative beliefs about oneself, others, or the world",
            "distorted blame of self or others for the event",
            "persistent negative emotional state",
            "diminished interest or participation in significant activities",
            "feelings of detachment or estrangement from others",
            "inability to experience positive emotions",
            "irritable behavior and angry outbursts",
            "reckless or self-destructive behavior",
            "hypervigilance",
            "exaggerated startle response",
            "problems with concentration",
            "sleep disturbance",
        ],
        "threshold": 6,
        "must_include_one_of": [],
        "duration_days": 30,
        "duration_note": "Duration of disturbance >1 month",
    },
    "Prolonged Grief Disorder": {
        "symptoms": [
            "intense yearning or longing for the deceased",
            "preoccupation with thoughts or memories of the deceased",
            "identity disruption since the death",
            "disbelief about the death",
            "avoidance of reminders of the loss",
            "intense emotional pain related to the death",
            "difficulty reintegrating into life",
            "emotional numbness",
            "feeling that life is meaningless",
            "intense loneliness",
        ],
        "threshold": 3,
        "must_include_one_of": ["intense yearning or longing for the deceased", "preoccupation with thoughts or memories of the deceased"],
        "duration_days": 365,
        "duration_note": "At least 12 months after bereavement in adults; 6 months in children",
    },
    "Acute Stress Disorder": {
        "symptoms": [
            "intrusive distressing memories",
            "recurrent distressing dreams",
            "dissociative reactions (flashbacks)",
            "intense psychological distress at trauma cues",
            "inability to experience positive emotions",
            "altered sense of reality",
            "inability to remember important aspects of the event",
            "efforts to avoid distressing memories or feelings",
            "efforts to avoid external reminders",
            "sleep disturbance",
            "irritable behavior",
            "hypervigilance",
            "problems with concentration",
            "exaggerated startle response",
        ],
        "threshold": 9,
        "must_include_one_of": [],
        "duration_days": 3,
        "duration_note": "3 days to 1 month after trauma exposure",
    },
    "Obsessive-Compulsive Disorder": {
        "symptoms": [
            "recurrent intrusive unwanted thoughts, urges, or images",
            "attempts to ignore, suppress, or neutralize obsessions",
            "repetitive behaviors or mental acts driven by obsessions",
            "behaviors aimed at preventing distress or dreaded events",
            "behaviors are excessive or not realistically connected to feared event",
        ],
        "threshold": 1,
        "must_include_one_of": [],
        "duration_days": None,
        "duration_note": "Time-consuming (≥1 hr/day) or cause clinically significant distress/impairment",
    },
    "Body Dysmorphic Disorder": {
        "symptoms": [
            "preoccupation with perceived defects in physical appearance not observable to others",
            "repetitive behaviors in response to appearance concerns (mirror checking, grooming, reassurance seeking)",
            "mental acts in response to appearance concerns (comparing appearance with others)",
        ],
        "threshold": 2,
        "must_include_one_of": ["preoccupation with perceived defects in physical appearance not observable to others"],
        "duration_days": None,
        "duration_note": "Persistent and chronic; typically 3-8 hours per day; no minimum duration specified",
    },
    "Hoarding Disorder": {
        "symptoms": [
            "persistent difficulty discarding possessions",
            "perceived need to save items",
            "distress associated with discarding",
            "accumulation congests and clutters living areas",
        ],
        "threshold": 3,
        "must_include_one_of": ["persistent difficulty discarding possessions"],
        "duration_days": None,
        "duration_note": "Persistent and long-standing; typically chronic, worsening each decade",
    },
    "Excoriation (Skin-Picking) Disorder": {
        "symptoms": [
            "recurrent skin picking resulting in skin lesions",
            "repeated attempts to decrease or stop skin picking",
        ],
        "threshold": 2,
        "must_include_one_of": ["recurrent skin picking resulting in skin lesions"],
        "duration_days": None,
        "duration_note": "Chronic; often months to years; no strict minimum duration",
    },
    "Trichotillomania (Hair-Pulling Disorder)": {
        "symptoms": [
            "recurrent pulling out of one's own hair resulting in hair loss",
            "repeated attempts to decrease or stop hair pulling",
        ],
        "threshold": 2,
        "must_include_one_of": ["recurrent pulling out of one's own hair resulting in hair loss"],
        "duration_days": None,
        "duration_note": "Chronic; may persist for months or years; no fixed minimum duration",
    },
    "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder": {
        "symptoms": [
            "obsessions predominating in clinical picture",
            "compulsions predominating in clinical picture",
            "skin picking behaviors",
            "hair pulling behaviors",
            "other body-focused repetitive behaviors",
        ],
        "threshold": 1,
        "must_include_one_of": [],
        "duration_days": None,
        "duration_note": "During or soon after substance intoxication/withdrawal; resolves within days to weeks",
    },
    "Bipolar I Disorder": {
        "symptoms": [
            "inflated self-esteem or grandiosity",
            "decreased need for sleep",
            "more talkative or pressure to keep talking",
            "flight of ideas or racing thoughts",
            "distractibility",
            "increase in goal-directed activity or psychomotor agitation",
            "excessive involvement in risky activities",
        ],
        "threshold": 3,
        "must_include_one_of": [],
        "duration_days": 7,
        "duration_note": "Manic episode ≥7 days (or any duration if hospitalization required); 4+ symptoms if mood only irritable",
    },
    "Bipolar II Disorder": {
        "symptoms": [
            "inflated self-esteem or grandiosity",
            "decreased need for sleep",
            "more talkative or pressure to keep talking",
            "flight of ideas or racing thoughts",
            "distractibility",
            "increase in goal-directed activity or psychomotor agitation",
            "excessive involvement in risky activities",
            "depressed mood",
            "anhedonia",
            "weight or appetite change",
            "sleep disturbance",
            "psychomotor agitation or retardation",
            "fatigue or loss of energy",
            "worthlessness or excessive guilt",
            "concentration difficulty or indecisiveness",
            "suicidal ideation or recurrent thoughts of death",
        ],
        "threshold": 3,
        "must_include_one_of": [],
        "duration_days": 4,
        "duration_note": "Hypomanic episode ≥4 days; Major depressive episode ≥2 weeks; no full manic episodes",
    },
    "Cyclothymic Disorder": {
        "symptoms": [
            "numerous periods of hypomanic symptoms not meeting hypomanic episode criteria",
            "numerous periods of depressive symptoms not meeting major depressive episode criteria",
            "symptoms present at least half the time",
            "not without symptoms for more than 2 months at a time",
        ],
        "threshold": 2,
        "must_include_one_of": [],
        "duration_days": 730,
        "duration_note": "At least 2 years in adults; 1 year in children/adolescents; never met criteria for manic, hypomanic, or major depressive episode",
    },
    "Substance/Medication-Induced Bipolar and Related Disorder": {
        "symptoms": [
            "abnormally elevated, expansive, or irritable mood",
            "abnormally increased activity or energy",
        ],
        "threshold": 2,
        "must_include_one_of": ["abnormally elevated, expansive, or irritable mood", "abnormally increased activity or energy"],
        "duration_days": None,
        "duration_note": "During or soon after substance intoxication/withdrawal/medication exposure; typically resolves within 1-2 days for stimulants",
    },
}


# ──────────────────────── SECTION C: EXCLUSIONS & DIFFERENTIALS ───────────────
# Extracted from CHUNK 5 (exclusion) and CHUNK 6 (differential) per disorder.

EXCLUSION_LINKS = [
    # (disorder, exclusion_type, excluding_condition)
    ("Major Depressive Disorder", "excludes_if", "Bipolar I Disorder"),
    ("Major Depressive Disorder", "excludes_if", "Bipolar II Disorder"),
    ("Major Depressive Disorder", "excludes_if", "Schizoaffective Disorder"),
    ("Major Depressive Disorder", "excludes_if", "substance/medication effects"),
    ("Major Depressive Disorder", "excludes_if", "another medical condition"),
    ("Persistent Depressive Disorder", "excludes_if", "Bipolar I Disorder"),
    ("Persistent Depressive Disorder", "excludes_if", "Bipolar II Disorder"),
    ("Persistent Depressive Disorder", "excludes_if", "Cyclothymic Disorder"),
    ("Substance/Medication-Induced Depressive Disorder", "excludes_if", "independent major depressive disorder"),
    ("Disruptive Mood Dysregulation Disorder", "excludes_if", "Bipolar I Disorder"),
    ("Disruptive Mood Dysregulation Disorder", "excludes_if", "Intermittent Explosive Disorder"),
    ("Disruptive Mood Dysregulation Disorder", "excludes_if", "Oppositional Defiant Disorder"),
    ("Separation Anxiety Disorder", "excludes_if", "substance/medication effects"),
    ("Separation Anxiety Disorder", "excludes_if", "another medical condition"),
    ("Specific Phobia", "excludes_if", "substance/medication effects"),
    ("Social Anxiety Disorder", "excludes_if", "substance/medication effects"),
    ("Panic Disorder", "excludes_if", "substance/medication effects"),
    ("Panic Disorder", "excludes_if", "another medical condition"),
    ("Posttraumatic Stress Disorder", "excludes_if", "substance/medication effects"),
    ("Posttraumatic Stress Disorder", "excludes_if", "another medical condition"),
    ("Acute Stress Disorder", "excludes_if", "Posttraumatic Stress Disorder"),
    ("Prolonged Grief Disorder", "excludes_if", "Major Depressive Disorder"),
    ("Obsessive-Compulsive Disorder", "excludes_if", "substance/medication effects"),
    ("Obsessive-Compulsive Disorder", "excludes_if", "another medical condition"),
    ("Body Dysmorphic Disorder", "excludes_if", "eating disorder (body fat/weight concerns)"),
    ("Hoarding Disorder", "excludes_if", "Obsessive-Compulsive Disorder"),
    ("Hoarding Disorder", "excludes_if", "neurodegenerative disorder"),
    ("Excoriation (Skin-Picking) Disorder", "excludes_if", "substance/medication effects"),
    ("Excoriation (Skin-Picking) Disorder", "excludes_if", "Body Dysmorphic Disorder"),
    ("Trichotillomania (Hair-Pulling Disorder)", "excludes_if", "Body Dysmorphic Disorder"),
    ("Trichotillomania (Hair-Pulling Disorder)", "excludes_if", "substance/medication effects"),
    ("Bipolar I Disorder", "excludes_if", "Schizoaffective Disorder"),
    ("Bipolar I Disorder", "excludes_if", "substance/medication effects"),
    ("Bipolar II Disorder", "excludes_if", "Bipolar I Disorder"),
    ("Bipolar II Disorder", "excludes_if", "Schizoaffective Disorder"),
    ("Cyclothymic Disorder", "excludes_if", "Bipolar I Disorder"),
    ("Cyclothymic Disorder", "excludes_if", "Bipolar II Disorder"),
    ("Cyclothymic Disorder", "excludes_if", "Schizoaffective Disorder"),
    ("Cyclothymic Disorder", "excludes_if", "substance/medication effects"),
    ("Substance/Medication-Induced Bipolar and Related Disorder", "excludes_if", "Bipolar I Disorder"),
    ("Substance/Medication-Induced Bipolar and Related Disorder", "excludes_if", "Bipolar II Disorder"),
]

DIFFERENTIAL_PAIRS = [
    # (disorder_a, disorder_b) — both directions are added in knowledge_graph.py
    ("Major Depressive Disorder", "Persistent Depressive Disorder"),
    ("Major Depressive Disorder", "Bipolar I Disorder"),
    ("Major Depressive Disorder", "Bipolar II Disorder"),
    ("Major Depressive Disorder", "Substance/Medication-Induced Depressive Disorder"),
    ("Major Depressive Disorder", "Depressive Disorder Due to Another Medical Condition"),
    ("Major Depressive Disorder", "Disruptive Mood Dysregulation Disorder"),
    ("Major Depressive Disorder", "Posttraumatic Stress Disorder"),
    ("Persistent Depressive Disorder", "Cyclothymic Disorder"),
    ("Social Anxiety Disorder", "Panic Disorder"),
    ("Social Anxiety Disorder", "Separation Anxiety Disorder"),
    ("Social Anxiety Disorder", "Selective Mutism"),
    ("Social Anxiety Disorder", "Body Dysmorphic Disorder"),
    ("Panic Disorder", "Social Anxiety Disorder"),
    ("Panic Disorder", "Specific Phobia"),
    ("Panic Disorder", "Posttraumatic Stress Disorder"),
    ("Obsessive-Compulsive Disorder", "Body Dysmorphic Disorder"),
    ("Obsessive-Compulsive Disorder", "Hoarding Disorder"),
    ("Obsessive-Compulsive Disorder", "Trichotillomania (Hair-Pulling Disorder)"),
    ("Obsessive-Compulsive Disorder", "Excoriation (Skin-Picking) Disorder"),
    ("Obsessive-Compulsive Disorder", "Social Anxiety Disorder"),
    ("Obsessive-Compulsive Disorder", "Major Depressive Disorder"),
    ("Body Dysmorphic Disorder", "Excoriation (Skin-Picking) Disorder"),
    ("Body Dysmorphic Disorder", "Trichotillomania (Hair-Pulling Disorder)"),
    ("Posttraumatic Stress Disorder", "Acute Stress Disorder"),
    ("Posttraumatic Stress Disorder", "Prolonged Grief Disorder"),
    ("Bipolar I Disorder", "Bipolar II Disorder"),
    ("Bipolar I Disorder", "Cyclothymic Disorder"),
    ("Bipolar I Disorder", "Major Depressive Disorder"),
    ("Bipolar II Disorder", "Cyclothymic Disorder"),
    ("Bipolar II Disorder", "Major Depressive Disorder"),
    ("Cyclothymic Disorder", "Bipolar I Disorder"),
    ("Cyclothymic Disorder", "Bipolar II Disorder"),
    ("Cyclothymic Disorder", "Substance/Medication-Induced Bipolar and Related Disorder"),
    ("Substance/Medication-Induced Bipolar and Related Disorder", "Bipolar I Disorder"),
    ("Substance/Medication-Induced Bipolar and Related Disorder", "Bipolar II Disorder"),
]
