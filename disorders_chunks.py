from langchain_core.documents import Document

chunks=[
Document(
        page_content= "## CHUNK 1 — OVERVIEW: Major Depressive Disorder (MDD)\n\nMajor depressive disorder represents the classic condition in the group of depressive disorders. It is characterized by discrete episodes involving clear-cut changes in affect, cognition, and neurovegetative functions. While the disorder is recurrent in the majority of cases, a diagnosis can be based on a single episode. \n\n**Core Features:**\n* Presence of at least one major depressive episode.\n* Absence of any history of manic or hypomanic episodes.\n* Essential symptoms include depressed mood or loss of interest/pleasure.\n* Represents a distinct change from the individual's previous functioning.",
    metadata= {
      "disorder": "Major Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "overview",
      "retrieval_priority": "high"
    }
  ),
 Document(
        page_content= "## CHUNK 2 — SYMPTOM LISt= Major Depressive Disorder\n\n**Diagnostic Threshold:** \nAt least **5 symptoms** must be present during the same 2-week period; at least one symptom must be either **(1),Depressed mood** or **(2),Loss of interest or pleasure**.\n\n**Symptoms:**\n1. **Depressed mood** most of the day, nearly every day (subjective or observed). Note: In children/adolescents, can be irritable mood.\n2. **Markedly diminished interest or pleasure** in all, or almost all, activities most of the day, nearly every day.\n3. **Significant weight change**: Loss or gain of >5% of body weight in a month, or daily appetite disturbance. Note: In children, consider failure to make expected weight gain.\n4. **Insomnia or hypersomnia** nearly every day.\n5. **Psychomotor agitation or retardation** nearly every day (must be observable by others).\n6. **Fatigue or loss of energy** nearly every day.\n7. **Feelings of worthlessness or excessive/inappropriate guilt** (may be delusional),nearly every day.\n8. **Diminished ability to think or concentrate**, or indecisiveness, nearly every day.\n9. **Recurrent thoughts of death**, recurrent suicidal ideation without a plan, a suicide attempt, or a specific plan.",
    metadata= {
      "disorder": "Major Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "symptom_list",
      "retrieval_priority": "high"
    }
  ),
   Document(
        page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Major Depressive Disorder\n\n* **Episode Duration:** Symptoms must persist for at least **2 weeks**.\n* **Persistence:** Symptoms must be present for most of the day, nearly every day during that 2-week period.\n* **Recurrence Interval:** For an episode to be considered recurrent, there must be an interval of at least **2 consecutive months** between separate episodes in which criteria for a major depressive episode are not met.",
    metadata= {
      "disorder": "Major Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "duration",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Major Depressive Disorder\n\n* The symptoms must cause **clinically significant distress** or **impairment** in social, occupational, or other important areas of functioning.\n* Symptoms must represent a clear change from the individual's previous level of functioning.",
    metadata= {
      "disorder": "Major Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "impairment",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Major Depressive Disorder\n\n* **Substance Exclusion:** The episode is not attributable to the physiological effects of a substance (drug of abuse or medication).\n* **Medical Condition Exclusion:** The symptoms are not attributable to the physiological effects of another medical condition.\n* **Bipolar Exclusion:** There has **never** been a manic episode or a hypomanic episode (unless substance-induced or medical).\n* **Psychotic Disorder Exclusion:** The episode is not better explained by schizoaffective disorder and is not superimposed on schizophrenia, schizophreniform disorder, delusional disorder, or other specified/unspecified schizophrenia spectrum and psychotic disorders.\n* **Symptom Attribute:** Do not include symptoms that are clearly attributable to another medical condition.",
    metadata= {
      "disorder": "Major Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "exclusion",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Major Depressive Disorder\n\n* **Normal Sadness and Grief:** Bereavement may induce great suffering but typically does not induce a major depressive episode. Clinical judgment is required to distinguish normal grief from MDD.\n* **Manic Episodes with Irritable Mood or Mixed Episodes:** MDD is distinguished by the absence of manic or hypomanic history.\n* **Disruptive Mood Dysregulation Disorder:** Characterized by chronic irritability rather than discrete episodes of depressed mood.\n* **Schizoaffective Disorder:** MDD is excluded if the symptoms are better explained by schizoaffective disorder or other psychotic disorders.\n* **Other Medical Conditions/Substances:** Distinguish MDD from depression-like phenomena caused by physiological effects of substances or medical illnesses.",
    metadata= {
      "disorder": "Major Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "differential",
      "retrieval_priority": "medium"
    }
   ),Document(
  
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Major Depressive Disorder\n\n* **Symptom Indicator:** Recurrent thoughts of death and suicidal ideation are core diagnostic criteria (Symptom 9).\n* **Specific Manifestations:** Includes suicidal ideation without a plan, specific plans for suicide, or suicide attempts.\n* **Clinical Concern:** MDD is associated with a risk of suicide attempts and completed suicide, which must be assessed during the diagnostic process.",
    metadata= {
      "disorder": "Major Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "suicide_risk",
      "retrieval_priority": "low"
    }
  ),Document(
    page_content= "## CHUNK 8 — COMORBIDITY: Major Depressive Disorder\n\n* **Substance-Related Disorders:** Frequently co-occurs with alcohol and drug use disorders; men are more likely to report comorbid alcohol and substance abuse.\n* **Panic Disorder:** Commonly co-occurs with major depressive disorder.\n* **Generalized Anxiety Disorder:** High rates of co-occurrence with MDD.\n* **Posttraumatic Stress Disorder (PTSD):** Frequently comorbid, especially following trauma exposure.\n* **Obsessive-Compulsive Disorder (OCD):** Notable co-occurrence with MDD.\n* **Anorexia Nervosa:** MDD frequently co-occurs with eating disorders including anorexia nervosa.\n* **Bulimia Nervosa:** Co-occurs with MDD; women are more likely to report comorbid bulimia nervosa.\n* **Borderline Personality Disorder:** High rates of comorbidity with MDD.\n* **Gender Differences in Comorbidity:** Women are more likely to report comorbid anxiety disorders, bulimia nervosa, and somatic symptom and related disorders; men are more likely to report comorbid alcohol and substance abuse.",
    metadata= {
      "disorder": "Major Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "comorbidity",
      "chunk_type": "comorbidity",
      "retrieval_priority": "high"
    }
  ),Document(
    page_content= "## CHUNK 1 — OVERVIEW: Persistent Depressive Disorder (F34.1)\n\nPersistent Depressive Disorder represents a consolidation of chronic major depressive disorder and dysthymic disorder.\n\n**Core Feature:**\n* Depressed mood for most of the day, for more days than not.\n* Duration of at least **2 years** in adults.\n* Duration of at least **1 year** in children and adolescents (mood may be irritable).\n\n**Course Characteristics:**\n* Chronic and persistent course.\n* Major depressive episodes may occur during the 2-year period.\n* If full criteria for a major depressive episode are met during the 2-year period, both diagnoses (Persistent Depressive Disorder and Major Depressive Disorder),are assigned.\n* During the required duration, symptom-free intervals must not exceed **2 months** at a time.",
    metadata= {
      "disorder": "Persistent Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "overview",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 2 — SYMPTOM LISt= Persistent Depressive Disorder\n\n**Diagnostic Threshold:**\n* Depressed mood (Criterion A),PLUS\n* At least **2 out of 6 symptoms** listed below\n\n**Mandatory Core Symptom (Required):**\n1. Depressed mood for most of the day, for more days than not (subjective or observed).\n   - In children/adolescents: mood may be irritable.\n\n**Additional Symptoms (At least 2 required):**\n2. Poor appetite or overeating.\n3. Insomnia or hypersomnia.\n4. Low energy or fatigue.\n5. Low self-esteem.\n6. Poor concentration or difficulty making decisions.\n7. Feelings of hopelessness.\n\n**Chronicity Rule:**\n* During the required 2-year period (1 year for youth), the individual has never been without the symptoms in Criteria A and B for more than **2 months at a time**.\n\n**Specifier Note:**\n* Full criteria for a major depressive episode may be continuously present for 2 years.\n* Specifiers include:\n  - With pure dysthymic syndrome\n  - With persistent major depressive episode\n  - With intermittent major depressive episodes (with or without current episode)",
    metadata= {
      "disorder": "Persistent Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "symptom_list",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Persistent Depressive Disorder\n\n* Symptoms must persist for at least **2 years** in adults.\n* In children and adolescents, duration must be at least **1 year**.\n* Depressed mood must be present for most of the day, for more days than not.\n* During the required duration, symptom-free periods must not exceed **2 consecutive months**.\n* Criteria for a major depressive disorder may be continuously present for the entire **2-year period**.",
    metadata= {
      "disorder": "Persistent Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "duration",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Persistent Depressive Disorder\n\n* The symptoms must cause **clinically significant distress** OR\n* Clinically significant **impairment** in social, occupational, or other important areas of functioning.\n* Functional impact may be as great as or greater than that of major depressive disorder.",
    metadata= {
      "disorder": "Persistent Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "impairment",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Persistent Depressive Disorder\n\n* **Manic/Hypomanic Exclusion:**\n  - There has never been a manic episode.\n  - There has never been a hypomanic episode.\n\n* **Psychotic Disorder Exclusion:**\n  - Not better explained by persistent schizoaffective disorder.\n  - Not better explained by schizophrenia.\n  - Not better explained by delusional disorder.\n  - Not better explained by other specified or unspecified schizophrenia spectrum and other psychotic disorders.\n\n* **Substance Exclusion:**\n  - Symptoms are not attributable to the physiological effects of a substance (drug of abuse, medication, toxin).\n\n* **Medical Condition Exclusion:**\n  - Symptoms are not attributable to another medical condition (e.g., hypothyroidism).\n\n* **Cyclothymic Disorder Exclusion:**\n  - If numerous hypomanic symptoms occur for at least 2 years without meeting criteria for major depressive, manic, or hypomanic episodes, diagnose cyclothymic disorder instead.\n\n* If full criteria for major depressive disorder are met during the 2-year period, diagnose major depressive disorder in addition to persistent depressive disorder.",
    metadata= {
      "disorder": "Persistent Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "exclusion",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Persistent Depressive Disorder\n\n* **Major Depressive Disorder:**\n  - If ≥2 years of depressed mood plus ≥2 Criterion B symptoms → Persistent Depressive Disorder.\n  - If full criteria for a major depressive episode are met at any time during this period → assign additional Major Depressive Disorder diagnosis.\n\n* **Other Specified or Unspecified Depressive Disorder:**\n  - Used if depressive symptoms persist ≥2 years but do not meet criteria for Persistent Depressive Disorder or Major Depressive Disorder.\n\n* **Bipolar I Disorder:**\n  - History of manic episode precludes diagnosis.\n\n* **Bipolar II Disorder:**\n  - History of hypomanic episode (without manic episode),with major depressive episodes.\n\n* **Cyclothymic Disorder:**\n  - ≥2 years of numerous hypomanic symptoms without meeting criteria for major depressive, manic, or hypomanic episodes.\n\n* **Psychotic Disorders:**\n  - No separate diagnosis if depressive symptoms occur exclusively during chronic psychotic disorder.\n\n* **Depressive or Bipolar Disorder Due to Another Medical Condition:**\n  - Diagnose if mood disturbance is directly attributable to medical condition.\n\n* **Substance/Medication-Induced Depressive or Bipolar Disorder:**\n  - Diagnose if substance is etiologically related.\n\n* **Personality Disorders:**\n  - May be diagnosed concurrently if criteria for both disorders are met.",
    metadata= {
      "disorder": "Persistent Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "differential",
      "retrieval_priority": "medium"
    }
   ),Document(
  
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Persistent Depressive Disorder\n\n* Associated with **elevated risk of suicidal outcomes**.\n* Elevated suicide risk is present across high-, middle-, and low-income countries.\n* Risk level is comparable to other chronic depressive conditions.\n* Assessment of suicidal thoughts or behaviors is clinically required due to chronic depressive course.",
    metadata= {
      "disorder": "Persistent Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "suicide_risk",
      "retrieval_priority": "low"
    }
 ),Document(
    page_content= "## CHUNK 8 — COMORBIDITY: Persistent Depressive Disorder\n\n* **Higher Comorbidity Risk:** Compared to individuals with major depressive disorder, those with persistent depressive disorder are at higher risk for psychiatric comorbidity in general.\n* **Anxiety Disorders:** Higher rates of comorbid anxiety disorders compared to MDD.\n* **Substance Use Disorders:** Elevated co-occurrence with alcohol and drug use disorders.\n* **Personality Disorders:** Particularly elevated risk for comorbid personality disorders.\n* **Cluster B Personality Disorders:** Early-onset persistent depressive disorder is strongly associated with DSM-5 Cluster B personality disorders (e.g., borderline, histrionic, antisocial, narcissistic).\n* **Cluster C Personality Disorders:** Early-onset persistent depressive disorder is also strongly associated with DSM-5 Cluster C personality disorders (e.g., avoidant, dependent, obsessive-compulsive personality disorder).",
    metadata= {
      "disorder": "Persistent Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "comorbidity",
      "chunk_type": "comorbidity",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 1 — OVERVIEW: Substance/Medication-Induced Depressive Disorder\n\nSubstance/Medication-Induced Depressive Disorder is characterized by a prominent and persistent mood disturbance caused by the direct physiological effects of a substance (drug of abuse, medication, or toxin).\n\n**Core Features:**\n* Prominent and persistent depressed mood OR\n* Markedly diminished interest or pleasure in all, or almost all, activities.\n* Symptoms must be etiologically related to substance intoxication, withdrawal, or medication exposure.\n\n**Clinical Rule:**\n* Diagnosis is made instead of substance intoxication or withdrawal only when depressive symptoms predominate and are severe enough to warrant independent clinical attention.\n\n**Temporal Relationship Requirement=**\n* Symptoms must develop during or soon after intoxication, withdrawal, or medication exposure/change.\n\n**Persistence Rule:**\n* If symptoms persist beyond approximately **4 weeks** after cessation of withdrawal/intoxication, consider independent depressive disorder.",
    metadata= {
      "disorder": "Substance/Medication-Induced Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "overview",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 2 — SYMPTOM LISt= Substance/Medication-Induced Depressive Disorder\n\n**Diagnostic Threshold:**\n* Presence of Criterion A mood disturbance\n* PLUS evidence of both B1 and B2\n\n**Criterion A — Required Mood Disturbance (Mandatory):**\n1. Depressed mood\n   OR\n2. Markedly diminished interest or pleasure in all, or almost all, activities\n\n**Criterion B — Substance Causation Evidence (Both Required):**\n3. Symptoms developed during or soon after:\n   - Substance intoxication\n   - Substance withdrawal\n   - Medication exposure\n   - Medication withdrawal\n4. The involved substance/medication is capable of producing depressive symptoms.\n\n**Criterion D Requirement=**\n5. Disturbance does not occur exclusively during delirium.\n\n**Specifier — Onset Type:**\n* With onset during intoxication\n* With onset during withdrawal\n* With onset after medication use",
    metadata= {
      "disorder": "Substance/Medication-Induced Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "symptom_list",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Substance/Medication-Induced Depressive Disorder\n\n* Symptoms must develop **during or soon after** substance intoxication, withdrawal, or medication exposure/change.\n* Most commonly onset occurs within the first **few weeks or within 1 month** of heavy substance use.\n* Depressive symptoms typically remit within **days to several weeks** after cessation.\n* If symptoms persist beyond approximately **4 weeks** after expected withdrawal course, evaluate for independent depressive disorder.",
    metadata= {
      "disorder": "Substance/Medication-Induced Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "duration",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Substance/Medication-Induced Depressive Disorder\n\n* The disturbance must cause **clinically significant distress** OR\n* Clinically significant **impairment** in social, occupational, or other important areas of functioning.\n* Symptoms must be sufficiently severe to warrant independent clinical attention beyond intoxication or withdrawal diagnosis.",
    metadata= {
      "disorder": "Substance/Medication-Induced Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "impairment",
      "retrieval_priority": "high"
    }
   ),Document(
      page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Substance/Medication-Induced Depressive Disorder\n\n* **Independent Depressive Disorder Exclusion:**\n  - Symptoms preceded substance/medication use.\n  - Symptoms persist for a substantial period (e.g., about **1 month**),after cessation of acute withdrawal or severe intoxication.\n  - History of recurrent non-substance-related depressive episodes.\n\n* **Delirium Exclusion:**\n  - Disturbance does not occur exclusively during delirium.\n\n* **Medical Condition Differential:**\n  - If mood disturbance is attributable to another medical condition, diagnose depressive disorder due to another medical condition instead.\n  - If both substance and medical condition contribute, both diagnoses may be assigned.\n\n* **Substance Intoxication/Withdrawal Rule:**\n  - Do not diagnose if symptoms are better explained solely by substance intoxication or withdrawal unless mood symptoms predominate and require independent attention.",
    metadata= {
      "disorder": "Substance/Medication-Induced Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "exclusion",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Substance/Medication-Induced Depressive Disorder\n\n* **Substance Intoxication and Withdrawal:**\n  - Depressive symptoms commonly occur during intoxication/withdrawal.\n  - Diagnose substance-induced depressive disorder only if mood symptoms predominate and are severe.\n\n* **Independent Depressive Disorder:**\n  - Diagnose if depressive syndrome occurs outside substance use periods.\n  - Diagnose if symptoms persist beyond expected withdrawal timeline (~4 weeks).\n\n* **Depressive Disorder Due to Another Medical Condition:**\n  - Diagnose if mood disturbance is due to direct physiological effects of a medical condition.\n  - If both medical condition and substance contribute, both diagnoses may be given.\n\n* **Other Specified or Unspecified Depressive Disorder:**\n  - Use if insufficient evidence to determine whether symptoms are substance-induced, medically induced, or independent.\n\n* **Comorbidity Consideration:**\n  - Substance use disorder may be recorded as mild, moderate, or severe within coding structure.\n  - Separate substance use disorder diagnosis is not assigned in addition to substance-induced depressive disorder; severity is incorporated into coding.",
    metadata= {
      "disorder": "Substance/Medication-Induced Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "differential",
      "retrieval_priority": "medium"
    }
   ),Document(
  
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Substance/Medication-Induced Depressive Disorder\n\n* Risk of suicide attempts is elevated among individuals with alcohol use disorder experiencing depressive episodes.\n* Increased suicide risk applies whether depressive episode is substance-induced or independent.\n* Suicide risk assessment is clinically required in individuals with co-occurring substance use and depressive symptoms.",
    metadata= {
      "disorder": "Substance/Medication-Induced Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "suicide_risk",
      "retrieval_priority": "low"
    }

   ),Document(
    page_content= "## CHUNK 8 — COMORBIDITY: Substance/Medication-Induced Depressive Disorder\n\n* **Higher Overall Psychiatric Comorbidity:** Individuals with substance/medication-induced depressive disorder have higher rates of comorbidity with any DSM mental disorder compared to those with independent major depressive disorder and no comorbid substance use disorder.\n* **Tobacco Use Disorder:** More likely to be present compared to independent MDD without substance use.\n* **Gambling Disorder:** Elevated co-occurrence compared to independent MDD.\n* **Antisocial Personality Disorder:** More likely comorbid compared to independent MDD without substance use disorder.\n* **Alcohol and Other Substance Use Disorders:** Compared to individuals with MDD and a comorbid substance use disorder, those with substance/medication-induced depressive disorder are more likely to have alcohol or other substance use disorder.\n* **Persistent Depressive Disorder:** Less likely to co-occur compared to both independent MDD (with or without substance use disorder).",
    metadata= {
      "disorder": "Substance/Medication-Induced Depressive Disorder",
      "category": "Depressive Disorders",
      "section": "comorbidity",
      "chunk_type": "comorbidity",
      "retrieval_priority": "high"
    }

   ),Document(
    page_content= "## CHUNK 1 — OVERVIEW: Disruptive Mood Dysregulation Disorder\n\nDisruptive Mood Dysregulation Disorder (DMDD),is characterized by chronic, severe, and persistent irritability with frequent temper outbursts.\n\n**Core Features:**\n* Severe recurrent temper outbursts (verbal and/or behavioral).\n* Persistent irritable or angry mood between outbursts.\n* Chronic (≥12 months),and non-episodic course.\n* Onset before age 10 years.\n\n**Age Parameters:**\n* Diagnosis should NOT be made before age 6 years.\n* Diagnosis should NOT be made after age 18 years.\n* Age at onset of Criteria A–E must be before 10 years.\n\n**Course Characteristic:**\n* Symptoms are persistent rather than episodic (distinguishes from bipolar disorders).",
    metadata= {
      "disorder": "Disruptive Mood Dysregulation Disorder",
      "category": "Depressive Disorders",
      "section": "overview",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 2 — SYMPTOM LISt= Disruptive Mood Dysregulation Disorder\n\n**Diagnostic Threshold:**\nAll Criteria A–D must be met.\n\n### Criterion A — Severe Temper Outbursts\n1. Severe recurrent temper outbursts manifested:\n   - Verbally (e.g., verbal rages)\n   - Behaviorally (e.g., physical aggression toward people or property)\n2. Outbursts are grossly out of proportion in intensity or duration to the situation or provocation.\n\n### Criterion B — Developmental Inconsistency\n3. Temper outbursts are inconsistent with developmental level.\n\n### Criterion C — Frequency Requirement\n4. Outbursts occur, on average, **≥3 times per week**.\n\n### Criterion D — Persistent Irritable Mood (Mandatory Core Feature)\n5. Mood between outbursts is persistently irritable or angry:\n   - Present most of the day\n   - Nearly every day\n   - Observable by others (e.g., parents, teachers, peers)\n\n### Criterion F — Setting Requirement\n6. Criteria A and D present in **≥2 of 3 settings**:\n   - Home\n   - School\n   - With peers\n7. Severe in at least **1 setting**.\n\n### Criterion I — Mania Exclusion Within Symptom Structure\n8. There has never been a distinct period lasting **>1 day** meeting full symptom criteria (except duration),for manic or hypomanic episode.",
    metadata= {
      "disorder": "Disruptive Mood Dysregulation Disorder",
      "category": "Depressive Disorders",
      "section": "symptom_list",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Disruptive Mood Dysregulation Disorder\n\n* Criteria A–D must be present for **≥12 months**.\n* During the 12-month period, there must NOT be a period of **≥3 consecutive months** without all symptoms (Criteria A–D).\n* Outbursts must occur on average **≥3 times per week**.\n* Onset of Criteria A–E must be before age **10 years**.",
    metadata= {
      "disorder": "Disruptive Mood Dysregulation Disorder",
      "category": "Depressive Disorders",
      "section": "duration",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Disruptive Mood Dysregulation Disorder\n\n* Symptoms must be present in **at least 2 settings** (home, school, peers).\n* Symptoms must be **severe in at least 1 setting**.\n* Clinical presentation results in marked disruption in:\n  - Family relationships\n  - Peer relationships\n  - School functioning\n* Symptoms must cause clinically significant impairment in functioning.",
    metadata= {
      "disorder": "Disruptive Mood Dysregulation Disorder",
      "category": "Depressive Disorders",
      "section": "impairment",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Disruptive Mood Dysregulation Disorder\n\n* **Bipolar Disorder Exclusion:**\n  - No history of manic or hypomanic episode lasting >1 day.\n  - If full-duration manic or hypomanic episode has occurred, do NOT assign DMDD.\n\n* **Major Depressive Episode Context Exclusion:**\n  - Symptoms do not occur exclusively during an episode of major depressive disorder.\n\n* **Other Mental Disorder Exclusion:**\n  - Not better explained by:\n    - Autism spectrum disorder\n    - Posttraumatic stress disorder\n    - Separation anxiety disorder\n    - Persistent depressive disorder\n\n* **Diagnostic Coexistence Rules:**\n  - Cannot coexist with oppositional defiant disorder.\n  - Cannot coexist with intermittent explosive disorder.\n  - Cannot coexist with bipolar disorder.\n  - If criteria met for both DMDD and oppositional defiant disorder → assign DMDD only.\n\n* **Substance/Medical Exclusion:**\n  - Symptoms not attributable to physiological effects of a substance.\n  - Symptoms not attributable to another medical or neurological condition.\n\n* **Contextual Exclusion:**\n  - Do not diagnose if irritability occurs only during anxiety-provoking situations.\n  - Do not diagnose if outbursts are secondary to autism spectrum disorder routine disruption.\n  - Do not diagnose if irritability occurs only during a major depressive episode.",
    metadata= {
      "disorder": "Disruptive Mood Dysregulation Disorder",
      "category": "Depressive Disorders",
      "section": "exclusion",
      "retrieval_priority": "high"
    }
   ),Document(
  
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Disruptive Mood Dysregulation Disorder\n\n* **Bipolar I and Bipolar II Disorders:**\n  - Bipolar disorders are episodic with distinct manic/hypomanic episodes.\n  - DMDD is chronic and non-episodic.\n  - Elevated/expansive mood and grandiosity are characteristic of mania but not DMDD.\n\n* **Oppositional Defiant Disorder (ODD):**\n  - ODD lacks persistent severe mood disturbance between outbursts.\n  - If criteria for both met → diagnose DMDD only.\n\n* **Intermittent Explosive Disorder (IED):**\n  - IED requires only 3 months of symptoms.\n  - IED does NOT require persistent irritable mood between outbursts.\n  - If persistent irritability present → diagnose DMDD only.\n\n* **Attention-Deficit/Hyperactivity Disorder (ADHD):**\n  - Can be comorbid.\n\n* **Major Depressive Disorder:**\n  - If irritability occurs only during depressive episode → diagnose MDD instead.\n\n* **Anxiety Disorder:**\n  - If irritability occurs only during anxiety exacerbation → diagnose anxiety disorder.\n\n* **Autism Spectrum Disorder:**\n  - If temper outbursts occur due to disrupted routines → diagnose autism spectrum disorder, not DMDD.",
    metadata= {
      "disorder": "Disruptive Mood Dysregulation Disorder",
      "category": "Depressive Disorders",
      "section": "differential",
      "retrieval_priority": "medium"
    }
   ),Document(
  
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Disruptive Mood Dysregulation Disorder\n\n* DMDD is associated with increased risk of developing:\n  - Unipolar depressive disorders in adulthood.\n  - Anxiety disorders in adulthood.\n* Chronic severe irritability is associated with significant psychosocial impairment.\n* Suicide risk assessment is clinically indicated due to elevated risk of later depressive disorders.",
    metadata= {
      "disorder": "Disruptive Mood Dysregulation Disorder",
      "category": "Depressive Disorders",
      "section": "suicide_risk",
      "retrieval_priority": "low"
    }
  ),Document(
    page_content= "## CHUNK 8 — COMORBIDITY: Disruptive Mood Dysregulation Disorder\n\n* **Extremely High Comorbidity Rates:** It is rare to find individuals whose symptoms meet criteria for DMDD alone; comorbidity rates are among the highest of any pediatric mental illness.\n* **Oppositional Defiant Disorder (ODD):** The strongest overlap is with ODD. However, if criteria for both ODD and DMDD are met, only DMDD is diagnosed.\n* **Intermittent Explosive Disorder (IED):** If criteria for both IED and DMDD are met, only DMDD is diagnosed.\n* **Bipolar Disorder:** Children with DMDD should not have symptoms meeting criteria for bipolar disorder; if they do, only the bipolar disorder diagnosis is assigned.\n* **Diverse Range of Comorbidities:** Children with DMDD typically present with a wide range of disruptive behavior, mood, anxiety, and autism spectrum symptoms and diagnoses.\n* **Anxiety Disorders:** Comorbid anxiety disorders are commonly observed.\n* **Autism Spectrum Disorder (ASD):** DMDD diagnosis should not be assigned if symptoms occur only when the routines of a child with ASD are disturbed.\n* **Obsessive-Compulsive Disorder (OCD):** DMDD diagnosis should not be assigned if symptoms occur only in the context of OCD routines being disturbed.\n* **Major Depressive Episode:** DMDD diagnosis should not be assigned if symptoms occur only in the context of a major depressive episode.",
    metadata= {
      "disorder": "Disruptive Mood Dysregulation Disorder",
      "category": "Depressive Disorders",
      "section": "comorbidity",
      "chunk_type": "comorbidity",
      "retrieval_priority": "high"
    }
  ),
Document(
page_content="## CHUNK 1 — OVERVIEW: Depressive Disorder Due to Another Medical Condition\n\n**Definition:**\nDepressive Disorder Due to Another Medical Condition is characterized by a **prominent and persistent disturbance in mood** that is judged to be the **direct physiological consequence of another medical condition**.\n\n**Core Clinical Features:**\n* Mood disturbance predominates in the clinical picture.\n* Mood disturbance must be etiologically linked to a **physiological mechanism of another medical illness**.\n* Diagnosis requires evidence from **history, physical examination, or laboratory findings** supporting the causal relationship.\n\n**Core Mood Presentations:**\n* Depressed mood.\n* Markedly diminished interest or pleasure in most or all activities (anhedonia).\n\n**Diagnostic Principle:**\n* The clinician must establish:\n  1. Presence of another medical condition.\n  2. Evidence that the depressive symptoms are a **direct pathophysiological consequence** of that condition.\n\n**Examples of Medical Conditions Associated With Depressive Symptoms:**\n* Cerebrovascular accident (CVA / stroke)\n* Parkinson’s disease\n* Huntington’s disease\n* Traumatic brain injury (TBI)\n* Hypothyroidism\n* Cushing’s syndrome\n* Systemic lupus erythematosus\n* Vitamin B12 deficiency\n* Multiple sclerosis\n* Pancreatic cancer\n\n**DSM Specifiers (ICD-10-CM Codes):**\n1. **F06.31 — With depressive features**\n   * Full criteria for Major Depressive Episode are **not met**.\n\n2. **F06.32 — With major depressive–like episode**\n   * Full criteria for Major Depressive Episode are **met (except exclusion criterion C for independent disorders)**.\n\n3. **F06.34 — With mixed features**\n   * Depressive symptoms present with **some manic or hypomanic symptoms**, but manic symptoms **do not predominate**.\n\n**Coding Rule:**\n* The **medical condition must be coded first**, followed by the depressive disorder diagnosis.\n* Example: \n  * E03.9 Hypothyroidism\n  * F06.31 Depressive disorder due to hypothyroidism, with depressive features.",
metadata={
"disorder": "Depressive Disorder Due to Another Medical Condition",
"category": "Depressive Disorders",
"section": "overview",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Depressive Disorder Due to Another Medical Condition\n\n**Core Diagnostic Requirement:**\nA **prominent and persistent mood disturbance** must predominate in the clinical picture.\n\n**Required Mood Symptoms (at least one must be present):**\n\n1. **Depressed mood**.\n\n2. **Markedly diminished interest or pleasure** in all, or almost all, activities (anhedonia).\n\n**Supporting Diagnostic Evidence (Required Criterion):**\n3. Evidence from **history, physical examination, or laboratory findings** demonstrating that the mood disturbance is the **direct physiological consequence of another medical condition**.\n\n**Important Diagnostic Clarification:**\n* DSM does **not specify a numeric symptom threshold** (e.g., ≥5 symptoms).\n* The diagnosis depends primarily on:\n  * Presence of **core depressive mood disturbance**, and\n  * Demonstrated **etiological link to a medical condition**.",
metadata={
"disorder": "Depressive Disorder Due to Another Medical Condition",
"category": "Depressive Disorders",
"section": "symptom_list",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Depressive Disorder Due to Another Medical Condition\n\n**Duration Rule:**\n* DSM does **not specify a fixed minimum duration (e.g., 2 weeks)**.\n\n**Temporal Relationship Requirement:**\nThe depressive disturbance must show a **temporal association with the medical condition**, such as:\n\n1. Onset of depressive symptoms occurring **after the onset of the medical illness**.\n2. Exacerbation of depressive symptoms occurring with **worsening of the medical condition**.\n3. Improvement or remission of depressive symptoms occurring with **treatment or remission of the medical condition**.\n\n**Course Considerations (Examples):**\n* Post-stroke depression may occur **within days to months after cerebrovascular accident**.\n* Depression may **precede neurological symptoms** in disorders such as Huntington’s disease.\n* Mood symptoms may be **episodic or recurrent** in some neurological disorders.",
metadata={
"disorder": "Depressive Disorder Due to Another Medical Condition",
"category": "Depressive Disorders",
"section": "duration",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Depressive Disorder Due to Another Medical Condition\n\n**Functional Impairment Criterion:**\n\n* The mood disturbance must cause **clinically significant distress** or **impairment** in one or more of the following areas:\n\n1. Social functioning\n2. Occupational functioning\n3. Other important areas of daily functioning\n\n**Diagnostic Clarification:**\n* The depressive symptoms must represent a **clinically meaningful disturbance** rather than a normal emotional reaction to illness alone.",
metadata={
"disorder": "Depressive Disorder Due to Another Medical Condition",
"category": "Depressive Disorders",
"section": "impairment",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Depressive Disorder Due to Another Medical Condition\n\nDiagnosis should **NOT be made** if the depressive disturbance is better explained by the following:\n\n**1. Other Mental Disorders**\n* The disturbance is better explained by another mental disorder such as:\n  * Adjustment disorder with depressed mood\n  * Independent depressive disorders\n\n**2. Delirium Exclusion**\n* The depressive disturbance **does not occur exclusively during the course of a delirium**.\n\n**3. Non-etiological Psychological Reaction**\n* Emotional responses to illness that reflect **psychological stress reactions rather than physiological causation** should not be diagnosed as this disorder.\n\n**4. Substance-Related Causes (Diagnostic Clarification)**\n* If depressive symptoms are primarily caused by **medications or substances**, the appropriate diagnosis may instead be **Substance/Medication-Induced Depressive Disorder**.\n\n**5. Diagnostic Attribution Rule**\n* The depressive symptoms must be attributable to the **physiological effects of the medical condition itself**, not solely the psychological impact of having the illness.",
metadata={
"disorder": "Depressive Disorder Due to Another Medical Condition",
"category": "Depressive Disorders",
"section": "exclusion",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Depressive Disorder Due to Another Medical Condition\n\n**1. Depressive Disorders Not Due to Medical Conditions**\nDiagnosis depends on evidence that:\n* Depressive symptoms **did not occur prior to the onset of the medical condition**.\n* The medical condition has **known potential to cause depressive symptoms**.\n* Depressive symptoms show **temporal relationship with onset or course of the medical illness**.\n\n**2. Medication-Induced Depressive Disorder**\n* Some medical conditions are treated with medications that can induce depression (e.g., steroids, interferon).\n* Clinical judgment is required to determine whether symptoms are due to:\n  * The medical condition itself, or\n  * A medication effect.\n\n**3. Delirium or Neurocognitive Disorders**\n* If depressive symptoms occur **only during delirium**, this diagnosis should **not** be made.\n* However, diagnosis may coexist with **major or mild neurocognitive disorder** if depressive symptoms are a physiological consequence of the underlying pathology.\n\n**4. Adjustment Disorder With Depressed Mood**\n* Adjustment disorder results from **psychological stress associated with illness**.\n* Differentiation depends on:\n  * **Pervasiveness of depressive symptoms**\n  * **Severity and number of depressive symptoms**.\n\n**5. Demoralization**\n* Demoralization involves:\n  * Helplessness\n  * Hopelessness\n  * Desire to give up\n* Typically **lacks anhedonia**, which is characteristic of depressive disorder.",
metadata={
"disorder": "Depressive Disorder Due to Another Medical Condition",
"category": "Depressive Disorders",
"section": "differential",
"retrieval_priority": "medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Depressive Disorder Due to Another Medical Condition\n\n**Suicide Risk Considerations:**\n\n* No epidemiological studies clearly differentiate suicide risk in this disorder from that in major depressive disorder.\n\n**Clinical Observations:**\n1. Case reports describe **suicides occurring during depressive episodes associated with medical conditions**.\n2. Serious medical illnesses themselves are associated with **increased suicide risk**, especially:\n   * Shortly after diagnosis\n   * Shortly after onset of illness\n\n**Clinical Interpretation:**\n* Suicide risk in depressive episodes associated with medical conditions should be assumed to be:\n  * **At least equal to**, or\n  * **Potentially greater than**, the risk seen in other major depressive episodes.\n\n**Clinical Implication:**\n* Assessment of **suicidal ideation, plans, and behavior** is necessary when evaluating individuals with this disorder.",
metadata={
"disorder": "Depressive Disorder Due to Another Medical Condition",
"category": "Depressive Disorders",
"section": "suicide_risk",
"retrieval_priority": "low"
}
)
,Document(
page_content="## CHUNK 8 — COMORBIDITY: Depressive Disorder Due to Another Medical Condition\n\n* **Medical Condition Comorbidities:** Comorbid conditions are those associated with the specific medical conditions of etiological relevance to the depressive disorder.\n* **Delirium:** Delirium can occur before or alongside depressive symptoms in individuals with a variety of medical conditions, such as Cushing's disease.\n* **Anxiety Symptoms:** The association of anxiety symptoms — usually generalized symptoms — is common in depressive disorders regardless of cause, including depressive disorder due to another medical condition.",
metadata={
"disorder": "Depressive Disorder Due to Another Medical Condition",
"category": "Depressive Disorders",
"section": "comorbidity",
"chunk_type": "comorbidity",
"retrieval_priority": "high"
}
)
,Document( page_content= "## CHUNK 1 — OVERVIEW: Separation Anxiety Disorder\n\nSeparation Anxiety Disorder is characterized by developmentally inappropriate and excessive fear or anxiety concerning separation from major attachment figures.\n\n**Core Feature:**\n* Excessive fear or anxiety about separation from home or major attachment figures.\n* Fear exceeds what is expected for the individual’s developmental level.\n* Can occur in children, adolescents, and adults.\n\n**Clinical Characteristics:**\n* Persistent distress when anticipating or experiencing separation.\n* Excessive worry about harm to attachment figures or events leading to separation.\n* Avoidance of situations involving separation.\n* May present with nightmares and physical symptoms when separation occurs or is anticipated.",
    metadata= {
      "disorder": "Separation Anxiety Disorder",
      "category": "Anxiety Disorder",
      "section": "overview",
      "retrieval_priority": "high"
    }
  ),Document(
    page_content= "## CHUNK 2 — SYMPTOM LISt= Separation Anxiety Disorder\n\n**Diagnostic Threshold:**\n* ≥3 symptoms required from the list below.\n\n**Mandatory Core Requirement=**\n* Developmentally inappropriate and excessive fear or anxiety concerning separation from major attachment figures.\n\n**Symptoms (At least 3 required):**\n1. Recurrent excessive distress when anticipating or experiencing separation from home or major attachment figures.\n2. Persistent and excessive worry about losing major attachment figures or about possible harm to them (e.g., illness, injury, disasters, death).\n3. Persistent and excessive worry about experiencing an untoward event (e.g., getting lost, being kidnapped, having an accident, becoming ill),that causes separation from a major attachment figure.\n4. Persistent reluctance or refusal to go out, away from home, to school, to work, or elsewhere because of fear of separation.\n5. Persistent and excessive fear of or reluctance about being alone or without major attachment figures at home or in other settings.\n6. Persistent reluctance or refusal to sleep away from home or to go to sleep without being near a major attachment figure.\n7. Repeated nightmares involving the theme of separation.\n8. Repeated complaints of physical symptoms (e.g., headaches, stomachaches, nausea, vomiting),when separation from major attachment figures occurs or is anticipated.",
    metadata= {
      "disorder": "Separation Anxiety Disorder",
      "category": "Anxiety Disorder",
      "section": "symptom_list",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Separation Anxiety Disorder\n\n* In children and adolescents (<18 years): Symptoms must persist for at least **4 weeks**.\n* In adults: Symptoms must persist typically for **6 months or more**.\n* Duration requirement for adults serves as a general guide with some flexibility.\n* Fear, anxiety, or avoidance must be persistent during the required duration.",
    metadata= {
      "disorder": "Separation Anxiety Disorder",
      "category": "Anxiety Disorder",
      "section": "duration",
      "retrieval_priority": "high"
    }
   ),Document(
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Separation Anxiety Disorder\n\n* The disturbance must cause **clinically significant distress** OR\n* Clinically significant **impairment** in social, academic, occupational, or other important areas of functioning.",
    metadata= {
      "disorder": "Separation Anxiety Disorder",
      "category": "Anxiety Disorder",
      "section": "impairment",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Separation Anxiety Disorder\n\n* **Autism Spectrum Disorder Exclusion:** Not better explained by refusal to leave home due to excessive resistance to change.\n* **Psychotic Disorder Exclusion:** Not better explained by delusions or hallucinations concerning separation.\n* **Agoraphobia Exclusion:** Not better explained by refusal to go outside without a trusted companion.\n* **Generalized Anxiety Disorder Exclusion:** Not better explained by worries about harm befalling significant others.\n* **Illness Anxiety Disorder Exclusion:** Not better explained by concerns about having an illness.\n* **Substance Exclusion:** Symptoms are not attributable to the physiological effects of a substance (drug of abuse, medication).\n* **Medical Condition Exclusion:** Symptoms are not attributable to another medical condition.",
    metadata= {
      "disorder": "Separation Anxiety Disorder",
      "category": "Anxiety Disorder",
      "section": "exclusion",
      "retrieval_priority": "high"
    }
  ),Document(
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Separation Anxiety Disorder\n\n* **Autism Spectrum Disorder:** Refusal to leave home due to resistance to change rather than separation fear.\n* **Psychotic Disorders:** Separation-related delusions or hallucinations.\n* **Agoraphobia=** Avoidance due to fear of panic-like symptoms or inability to escape.\n* **Generalized Anxiety Disorder:** Excessive worry about multiple domains rather than separation-specific fear.\n* **Illness Anxiety Disorder:** Preoccupation with having or acquiring a serious illness rather than separation fear.\n\n* Differentiation is based on the primary feared situation and associated cognitions.",
    metadata= {
      "disorder": "Separation Anxiety Disorder",
      "category": "Anxiety Disorder",
      "section": "differential",
      "retrieval_priority": "medium"
    }
  ),Document(
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Separation Anxiety Disorder\n\n* Individuals with anxiety disorders may be more likely to experience suicidal thoughts.\n* Anxiety disorders are associated with increased risk of suicide attempts compared to individuals without anxiety disorders.\n* Panic disorder, generalized anxiety disorder, and specific phobia show stronger associations with transition from suicidal ideation to attempt.\n* Suicide risk assessment should be conducted when clinically indicated.",
    metadata= {
      "disorder": "Separation Anxiety Disorder",
      "category": "Anxiety Disorder",
      "section": "suicide_risk",
      "retrieval_priority": "low"
    }
  ),Document(
    page_content= "## CHUNK 8 — COMORBIDITY: Separation Anxiety Disorder\n\n* **Children — Most Common Comorbidities:**\n  - Generalized Anxiety Disorder: Highly comorbid with separation anxiety disorder in children.\n  - Specific Phobia: Highly comorbid with separation anxiety disorder in children.\n\n* **Adults — Common Comorbidities:**\n  - Specific Phobia\n  - Posttraumatic Stress Disorder (PTSD)\n  - Panic Disorder\n  - Generalized Anxiety Disorder\n  - Social Anxiety Disorder\n  - Agoraphobia\n  - Obsessive-Compulsive Disorder (OCD)\n  - Prolonged Grief Disorder\n  - Personality Disorders (particularly Cluster C: dependent, avoidant, and obsessive-compulsive personality disorders)\n\n* **Mood Disorders (Adults):** Depressive disorders and bipolar disorders are also comorbid with separation anxiety disorder in adults.",
    metadata= {
      "disorder": "Separation Anxiety Disorder",
      "category": "Anxiety Disorder",
      "section": "comorbidity",
      "chunk_type": "comorbidity",
      "retrieval_priority": "high"
    }

),
Document(
    page_content= "## CHUNK 1 — OVERVIEW: Specific Phobia\n\nSpecific Phobia is characterized by marked fear or anxiety about a specific object or situation (phobic stimulus).\n\n**Core Feature:**\n* Fear or anxiety is circumscribed to a specific object or situation.\n* Fear response is immediate upon exposure.\n* Object or situation is actively avoided or endured with intense fear.\n\n**Common Phobic Stimuli (Specifiers):**\n* Animal (e.g., spiders, dogs).\n* Natural environment (e.g., heights, storms, water).\n* Blood-injection-injury (e.g., needles, medical procedures).\n* Situational (e.g., airplanes, elevators, enclosed places).\n* Other (e.g., choking, vomiting, loud sounds in children).\n\n**Clinical Characteristics:**\n* Fear is out of proportion to actual danger.\n* Recognition of disproportionality may be present.\n* Frequently begins in childhood.",
    metadata= {
      "disorder": "Specific Phobia",
      "category": "Anxiety Disorder",
      "section": "overview",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 2 — SYMPTOM LISt= Specific Phobia\n\n**Diagnostic Threshold:**\n* All criteria A–G must be met.\n\n**Symptoms:**\n1. Marked fear or anxiety about a specific object or situation (e.g., flying, heights, animals, injections, blood).\n   - In children: may be expressed by crying, tantrums, freezing, or clinging.\n2. The phobic object or situation almost always provokes immediate fear or anxiety.\n3. The phobic object or situation is actively avoided OR endured with intense fear or anxiety.\n4. The fear or anxiety is out of proportion to the actual danger posed and to the sociocultural context.\n5. The fear, anxiety, or avoidance is persistent.\n6. The fear, anxiety, or avoidance causes clinically significant distress or impairment.\n7. The disturbance is not better explained by another mental disorder.\n\n**Specifier Requirement=**\n* Code based on phobic stimulus (Animal, Natural environment, Blood-injection-injury, Situational, Other).",
    metadata= {
      "disorder": "Specific Phobia",
      "category": "Anxiety Disorder",
      "section": "symptom_list",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Specific Phobia\n\n* Symptoms must persist for **≥6 months**.\n* Persistence applies to fear, anxiety, or avoidance.\n* Duration criterion helps distinguish disorder from transient developmental fears.",
    metadata= {
      "disorder": "Specific Phobia",
      "category": "Anxiety Disorder",
      "section": "duration",
      "retrieval_priority": "high"}
 ),Document(
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Specific Phobia\n\n* The fear, anxiety, or avoidance must cause:\n  - **Clinically significant distress**, OR\n  - **Clinically significant impairment** in social, occupational, or other important areas of functioning.\n\n* Degree of impairment may increase with number of feared objects or situations.",
    metadata= {
      "disorder": "Specific Phobia",
      "category": "Anxiety Disorder",
      "section": "impairment",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Specific Phobia\n\n* **Agoraphobia Exclusion:** Not better explained by fear of situations due to difficulty escaping or lack of help during panic-like symptoms.\n* **Social Anxiety Disorder Exclusion:** Not better explained by fear of negative evaluation.\n* **Separation Anxiety Disorder Exclusion:** Not better explained by fear of separation from attachment figures.\n* **Panic Disorder Exclusion:** Not better explained by recurrent unexpected panic attacks.\n* **Obsessive-Compulsive Disorder Exclusion:** Not due to obsessions (e.g., contamination fears).\n* **Posttraumatic Stress Disorder Exclusion:** Not exclusively due to trauma reminders when PTSD criteria are met.\n* **Eating Disorder Exclusion:** Not limited to food avoidance due to anorexia nervosa or bulimia nervosa.\n* **Psychotic Disorder Exclusion:** Not attributable to delusional beliefs.\n* **Substance Exclusion:** Not attributable to physiological effects of a substance/medication.\n* **Medical Condition Exclusion:** Not attributable to another medical condition.",
    metadata= {
      "disorder": "Specific Phobia",
      "category": "Anxiety Disorder",
      "section": "exclusion",
      "retrieval_priority": "high"
    }
   ),Document(
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Specific Phobia\n\n* **Agoraphobia=** Multiple feared situations with concern about escape/help availability.\n* **Social Anxiety Disorder:** Fear based on negative evaluation by others.\n* **Separation Anxiety Disorder:** Fear due to separation from attachment figures.\n* **Panic Disorder:** Unexpected panic attacks occurring outside specific phobic exposure.\n* **Obsessive-Compulsive Disorder:** Avoidance driven by obsessions.\n* **Posttraumatic Stress Disorder:** Fear related to trauma reminders with full PTSD criteria.\n* **Eating Disorders:** Avoidance limited to food-related stimuli.\n* **Schizophrenia Spectrum and Other Psychotic Disorders:** Fear attributable to delusional thinking.\n\n* Differentiation is based on primary feared stimulus, cognition, and panic pattern.",
    metadata= {
      "disorder": "Specific Phobia",
      "category": "Anxiety Disorder",
      "section": "differential",
      "retrieval_priority": "medium"
    }
  ),Document(
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Specific Phobia\n\n* Specific phobia is associated with suicidal thoughts.\n* Specific phobia is associated with suicide attempts.\n* Specific phobia is associated with transition from suicidal ideation to attempt.\n* Increased risk observed in epidemiological and prospective studies.",
    metadata= {
      "disorder": "Specific Phobia",
      "category": "Anxiety Disorder",
      "section": "suicide_risk",
      "retrieval_priority": "low"
    }
),Document(
    page_content= "## CHUNK 8 — COMORBIDITY: Specific Phobia\n\n* **Rarely Seen Alone:** Specific phobia is rarely seen in medical-clinical settings in the absence of other psychopathology; more frequently seen in non-medical mental health settings.\n* **Other Anxiety Disorders:** Most commonly associated comorbid condition; specific phobia is typically the temporally primary disorder due to early onset.\n* **Depressive and Bipolar Disorders:** Individuals with specific phobia are at increased risk for developing depressive and bipolar disorders.\n* **Substance-Related Disorders:** Elevated co-occurrence; may represent self-medication.\n* **Somatic Symptom and Related Disorders:** Noted comorbidity.\n* **Personality Disorders:** Particularly dependent personality disorder.",
    metadata= {
      "disorder": "Specific Phobia",
      "category": "Anxiety Disorder",
      "section": "comorbidity",
      "chunk_type": "comorbidity",
      "retrieval_priority": "high"
    }
)
 ,Document(
    page_content= "## CHUNK 1 — OVERVIEW: Selective Mutism\n\nSelective Mutism is an anxiety disorder characterized by a consistent failure to speak in specific social situations where there is an expectation to speak, despite speaking in other situations.\n\n**Core Feature:**\n- Child speaks normally in comfortable settings (e.g., home).\n- Child fails to speak in certain social settings (e.g., school).\n\n**Key Pattern:**\n- Not due to lack of language knowledge.\n- Not due to communication disorder.\n- Strongly associated with social anxiety.\n\n**Typical Onset=**\n- Usually before age 5.\n- Often first noticed when school begins.\n\n**Category:** Anxiety Disorder.",
    metadata= {
      "disorder": "Selective Mutism",
      "category": "Anxiety Disorder",
      "section": "overview",
      "retrieval_priority": "high"
}
 ),Document(
    page_content= "## CHUNK 2 — SYMPTOM LISt= Selective Mutism\n\n**All criteria must be met=**\n\n1. Consistent failure to speak in specific social situations where speaking is expected (e.g., school), despite speaking in other situations.\n2. The disturbance interferes with educational or occupational achievement or with social communication.\n3. Duration is at least 1 month (not limited to first month of school).\n4. Failure to speak is not due to lack of knowledge of, or comfort with, the spoken language required.\n5. Not better explained by a communication disorder.\n6. Does not occur exclusively during Autism Spectrum Disorder, Schizophrenia Spectrum, or other psychotic disorders.",
    metadata= {
      "disorder": "Selective Mutism",
      "category": "Anxiety Disorder",
      "section": "symptom_list",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 3 — DURATION REQUIREMENt= Selective Mutism\n\n- Symptoms must last ≥ 1 month.\n- Not limited to the first month of school (to avoid mislabeling normal adjustment).",
    metadata= {
      "disorder": "Selective Mutism",
      "category": "Anxiety Disorder",
      "section": "duration",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Selective Mutism\n\n- Causes clinically significant interference in:\n  - Educational functioning, OR\n  - Occupational functioning, OR\n  - Social communication.\n\n- Child may use gestures, whispering, or nonverbal communication instead of speech.",
    metadata= {
      "disorder": "Selective Mutism",
      "category": "Anxiety Disorder",
      "section": "impairment",
      "retrieval_priority": "high"
    }
 ),Document(
    page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Selective Mutism\n\n- Not due to lack of knowledge of the spoken language.\n- Not due to limited language proficiency in bilingual children.\n- Not better explained by:\n  - Communication disorders (e.g., childhood-onset fluency disorder).\n  - Autism Spectrum Disorder.\n  - Schizophrenia Spectrum disorders.\n  - Other psychotic disorders.\n- Not attributable to substance use or medical condition.",
    metadata= {
      "disorder": "Selective Mutism",
      "category": "Anxiety Disorder",
      "section": "exclusion",
      "retrieval_priority": "high"
    }
  ),Document(
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Selective Mutism\n\n- **Social Anxiety Disorder:** Child speaks but with intense fear; in selective mutism, there is consistent silence in specific settings.\n- **Autism Spectrum Disorder:** Broader social communication deficits and restricted behaviors.\n- **Communication Disorders:** Speech impairment present across settings.\n- **Psychotic Disorders:** Mutism may be due to disorganized or psychotic symptoms.\n\n- Primary differentiation: situation-specific failure to speak despite normal speech ability elsewhere.",
    metadata= {
      "disorder": "Selective Mutism",
      "category": "Anxiety Disorder",
      "section": "differential",
      "retrieval_priority": "medium"
    }
 ),Document(
    page_content= "## CHUNK 7 — COMORBIDITY: Selective Mutism\n\n* **Other Anxiety Disorders:** The most common comorbid conditions; particularly social anxiety disorder, followed by separation anxiety disorder and specific phobia.\n* **Autism Spectrum Disorder:** Frequently co-occurs with selective mutism in clinical settings.\n* **Oppositional Behaviors:** Present in a substantial minority of children; may be limited to situations requiring speech.\n* **Communication Delays or Disorders:** May appear in some children with selective mutism.",
    metadata= {
      "disorder": "Selective Mutism",
      "category": "Anxiety Disorder",
      "section": "comorbidity",
      "chunk_type": "comorbidity",
      "retrieval_priority": "high"
    }
 ),
Document(
 page_content= "## CHUNK 1 — OVERVIEW: Social Anxiety Disorder\n\nSocial Anxiety Disorder is characterized by marked fear or anxiety about one or more social situations in which the individual is exposed to possible scrutiny by others.\n\nCore Feature:\n- Intense fear of negative evaluation, humiliation, embarrassment, rejection, or offending others.\n- Social situations almost always provoke fear or anxiety.\n- Situations are avoided or endured with intense distress.\n\nCommon Social Situations:\n- Social interactions (e.g., conversations, meeting unfamiliar people).\n- Being observed (e.g., eating, drinking).\n- Performing in front of others (e.g., giving a speech).\n\nIn children:\n- Anxiety must occur in peer settings, not just with adults.\n- May be expressed as crying, tantrums, freezing, clinging, shrinking, or failing to speak.\n\nSpecifier:\n- Performance Only: Fear restricted to speaking or performing in public.\n\nCategory: Anxiety Disorder.",
 metadata= {
  "disorder": "Social Anxiety Disorder",
  "category": "Anxiety Disorder",
  "section": "overview",
  "retrieval_priority": "high"
 }
),

Document(
 page_content= "## CHUNK 2 — SYMPTOM LISt= Social Anxiety Disorder\n\nAll criteria A–J must be met=\n\nA. Marked fear or anxiety about one or more social situations involving possible scrutiny.\nB. Fear of acting in a way or showing anxiety symptoms that will be negatively evaluated.\nC. Social situations almost always provoke fear or anxiety.\nD. Social situations are avoided or endured with intense fear or anxiety.\nE. Fear or anxiety is out of proportion to actual threat and sociocultural context.\nF. Fear, anxiety, or avoidance is persistent (typically ≥6 months).\nG. Causes clinically significant distress or impairment in social, occupational, or other important areas.\nH. Not attributable to substance use or another medical condition.\nI. Not better explained by another mental disorder.\nJ. If another medical condition is present, fear is clearly unrelated or excessive.\n\nPerformance Only Specifier:\n- Fear restricted to speaking or performing in public.",
 metadata= {
  "disorder": "Social Anxiety Disorder",
  "category": "Anxiety Disorder",
  "section": "symptom_list",
  "retrieval_priority": "high"
 }
),

Document(
 page_content= "## CHUNK 3 — DURATION REQUIREMENt= Social Anxiety Disorder\n\n- Fear, anxiety, or avoidance is persistent.\n- Duration is typically ≥6 months.\n- Criterion helps distinguish disorder from transient social fears.\n\n- Anticipatory anxiety may occur days or weeks before social events.",
 metadata= {
  "disorder": "Social Anxiety Disorder",
  "category": "Anxiety Disorder",
  "section": "duration",
  "retrieval_priority": "high"
 }
),

Document(
 page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Social Anxiety Disorder\n\n- Causes clinically significant distress, OR\n- Causes clinically significant impairment in:\n   - Social functioning\n   - Occupational functioning\n   - Academic performance\n   - Relationships\n\nFunctional Consequences:\n- School dropout\n- Reduced workplace productivity\n- Lower quality of life\n- Social isolation\n- Delayed life milestones (e.g., marriage, career advancement)",
 metadata= {
  "disorder": "Social Anxiety Disorder",
  "category": "Anxiety Disorder",
  "section": "impairment",
  "retrieval_priority": "high"
 }
),

Document(
 page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Social Anxiety Disorder\n\n- Not attributable to physiological effects of a substance (e.g., drugs, medication).\n- Not attributable to another medical condition.\n- Not better explained by another mental disorder, including:\n   - Panic Disorder (panic attacks not uncued)\n   - Agoraphobia (fear of escape difficulty rather than scrutiny)\n   - Generalized Anxiety Disorder (worries broader than evaluation)\n   - Separation Anxiety Disorder\n   - Body Dysmorphic Disorder (appearance-focused fear only)\n   - Autism Spectrum Disorder\n   - Delusional Disorder\n   - Eating Disorders (if fear limited to eating behaviors)\n   - Obsessive-Compulsive Disorder (if limited to obsessions/compulsions)\n\n- If medical condition (e.g., Parkinson’s disease, obesity, disfigurement),is present, fear must be excessive or unrelated.",
 metadata= {
  "disorder": "Social Anxiety Disorder",
  "category": "Anxiety Disorder",
  "section": "exclusion",
  "retrieval_priority": "high"
 }
)
,
Document(
 page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Social Anxiety Disorder\n\nNormative Shyness:\n- Common personality trait without significant impairment.\n\nAgoraphobia=\n- Fear of being unable to escape or get help during panic-like symptoms.\n\nPanic Disorder:\n- Panic attacks are unexpected and uncued.\n\nGeneralized Anxiety Disorder:\n- Worries are broad and not limited to social evaluation.\n\nSeparation Anxiety Disorder:\n- Fear of separation from attachment figures.\n\nSelective Mutism:\n- Failure to speak limited to certain settings.\n\nMajor Depressive Disorder:\n- Negative self-view rather than fear of evaluation.\n\nBody Dysmorphic Disorder:\n- Social fear limited to perceived appearance defects.\n\nAutism Spectrum Disorder:\n- Broader social communication deficits and restricted behaviors.\n\nAvoidant Personality Disorder:\n- More pervasive avoidance pattern and negative self-concept.\n\nPsychotic Disorders:\n- Social fears driven by delusional beliefs.\n\nDifferentiation is based on primary feared stimulus, cognition pattern, and presence of impairment.",
 metadata= {
  "disorder": "Social Anxiety Disorder",
  "category": "Anxiety Disorder",
  "section": "differential",
  "retrieval_priority": "medium"
 }
)
,
Document(
 page_content= "## CHUNK 7 — ASSOCIATION WITH SUICIDAL THOUGHTS OR BEHAVIOR: Social Anxiety Disorder\n\n- Associated with increased risk of suicidal ideation.\n- Associated with increased risk of suicide attempts.\n- Risk observed independently of major depressive disorder in some populations.\n- Chronic social isolation may increase depression risk, which further elevates suicide risk.",
 metadata= {
  "disorder": "Social Anxiety Disorder",
  "category": "Anxiety Disorder",
  "section": "suicide_risk",
  "retrieval_priority": "low"
 }
),Document(
 page_content= "## CHUNK 8 — COMORBIDITY: Social Anxiety Disorder\n\n* **Other Anxiety Disorders:** Frequently comorbid; onset of social anxiety disorder generally precedes other anxiety disorders, except specific phobia and separation anxiety disorder.\n* **Major Depressive Disorder:** High comorbidity rate; chronic social isolation from social anxiety disorder may result in major depressive disorder. Comorbidity with depression is also high in older adults.\n* **Substance Use Disorders:** Substances may be used as self-medication for social fears; however, substance intoxication or withdrawal symptoms (e.g., trembling) may further increase social fear.\n* **Body Dysmorphic Disorder:** Frequently comorbid with social anxiety disorder.\n* **Avoidant Personality Disorder:** Generalized social anxiety disorder is often comorbid with avoidant personality disorder.\n* **Autism Spectrum Disorder (High-Functioning):** Common comorbidity in children.\n* **Selective Mutism:** Common comorbidity in children.",
 metadata= {
  "disorder": "Social Anxiety Disorder",
  "category": "Anxiety Disorder",
  "section": "comorbidity",
  "chunk_type": "comorbidity",
  "retrieval_priority": "high"
 }
),
Document(
 page_content= "## CHUNK 1 — OVERVIEW: Panic Disorder (F41.0)\n\nPanic Disorder is characterized by recurrent unexpected panic attacks and persistent concern or behavioral change related to the attacks.\n\nCore Feature:\n- Recurrent unexpected panic attacks (abrupt surge of intense fear or discomfort peaking within minutes).\n- At least one attack followed by ≥1 month of worry or maladaptive behavioral change.\n\nPanic Attack Definition:\n- Sudden surge of intense fear/discomfort.\n- Peaks within minutes.\n- Occurs from calm or anxious state.\n- Unexpected = no obvious trigger (“out of the blue”).\n\nAttacks may be:\n- Daytime or nocturnal (waking from sleep in panic).\n- Full-symptom (≥4 symptoms),or limited-symptom (<4).\n\nCategory: Anxiety Disorder.",
 metadata= {
  "disorder": "Panic Disorder",
  "category": "Anxiety Disorder",
  "section": "overview",
  "retrieval_priority": "high"
 }
),

Document(
 page_content= "## CHUNK 2 — SYMPTOM LISt= Panic Disorder\n\nA. Recurrent unexpected panic attacks. During an attack, four (or more),of the following symptoms occur:\n1. Palpitations, pounding heart, or accelerated heart rate.\n2. Sweating.\n3. Trembling or shaking.\n4. Shortness of breath or smothering sensations.\n5. Feelings of choking.\n6. Chest pain or discomfort.\n7. Nausea or abdominal distress.\n8. Dizziness, unsteadiness, light-headedness, or faintness.\n9. Chills or heat sensations.\n10. Paresthesias (numbness or tingling).\n11. Derealization or depersonalization.\n12. Fear of losing control or “going crazy.”\n13. Fear of dying.\n\nNote: Culture-specific symptoms (e.g., tinnitus, screaming),do not count toward the four required symptoms.\n\nB. At least one attack followed by ≥1 month of:\n   1. Persistent concern or worry about additional attacks or consequences, OR\n   2. Significant maladaptive behavioral change related to attacks (e.g., avoiding exercise or unfamiliar situations).\n\nC. Not attributable to substance use or another medical condition.\nD. Not better explained by another mental disorder.",
 metadata= {
  "disorder": "Panic Disorder",
  "category": "Anxiety Disorder",
  "section": "symptom_list",
  "retrieval_priority": "high"
 }
)
,
Document(
 page_content= "## CHUNK 3 — DURATION REQUIREMENt= Panic Disorder\n\n- At least one panic attack must be followed by ≥1 month of:\n   - Persistent worry about additional attacks or consequences, OR\n   - Maladaptive behavioral change related to attacks.\n\n- Disorder often follows a chronic, waxing-and-waning course if untreated.",
 metadata= {
  "disorder": "Panic Disorder",
  "category": "Anxiety Disorder",
  "section": "duration",
  "retrieval_priority": "high"
 }
)
,
Document(
 page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENt= Panic Disorder\n\n- Causes clinically significant distress, OR\n- Causes impairment in social, occupational, or other important areas.\n\nFunctional Consequences:\n- Frequent emergency or medical visits.\n- Work or school absenteeism.\n- Social avoidance.\n- Reduced quality of life.\n- Economic burden.\n\n- Higher morbidity associated with full-symptom attacks.",
 metadata= {
  "disorder": "Panic Disorder",
  "category": "Anxiety Disorder",
  "section": "impairment",
  "retrieval_priority": "high"
 }
)
,
Document(
 page_content= "## CHUNK 5 — EXCLUSION CRITERIa= Panic Disorder\n\n- Not due to physiological effects of a substance (e.g., stimulant intoxication, alcohol withdrawal).\n- Not due to another medical condition (e.g., hyperthyroidism, arrhythmias, asthma, seizure disorders).\n- Not better explained by another mental disorder, including:\n   - Social Anxiety Disorder (attacks only in social situations).\n   - Specific Phobia (triggered by phobic object).\n   - Agoraphobia (fear of escape difficulty).\n   - Generalized Anxiety Disorder (triggered by worry).\n   - Separation Anxiety Disorder (triggered by separation).\n   - Obsessive-Compulsive Disorder (triggered by obsessions).\n   - Posttraumatic Stress Disorder (triggered by trauma reminders).\n\n- If panic attacks occur only in response to specific triggers, diagnose the relevant disorder instead.\n- If unexpected attacks plus persistent concern/behavioral change are present, Panic Disorder may be diagnosed in addition.",
 metadata= {
  "disorder": "Panic Disorder",
  "category": "Anxiety Disorder",
  "section": "exclusion",
  "retrieval_priority": "high"
 }
)
,
Document(
 page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Panic Disorder\n\nOnly Limited-Symptom Attacks:\n- If full unexpected panic attacks never occurred, consider other specified or unspecified anxiety disorder.\n\nAnxiety Disorder Due to Another Medical Condition:\n- Panic symptoms directly caused by medical illness.\n\nSubstance/Medication-Induced Anxiety Disorder:\n- Panic symptoms directly caused by intoxication or withdrawal.\n\nOther Anxiety Disorder:\n- Panic attacks are expected and triggered by specific fears.\n\nPsychotic Disorders:\n- Panic symptoms occur alongside delusions or hallucinations.\n\nDifferentiation is based on unexpected nature of attacks and presence of persistent worry or behavioral change.",
 metadata= {
  "disorder": "Panic Disorder",
  "category": "Anxiety Disorder",
  "section": "differential",
  "retrieval_priority": "medium"
 }
),

Document(
 page_content= "## CHUNK 7 — ASSOCIATION WITH SUICIDAL THOUGHTS OR BEHAVIOR: Panic Disorder\n\n- Panic disorder is associated with increased suicidal ideation.\n- Associated with increased suicide attempts.\n- Risk remains elevated even after controlling for comorbid conditions.\n- Approximately 25% of primary care patients with panic disorder report suicidal thoughts.\n- Cognitive symptoms (e.g., derealization),may be linked to suicidal thoughts; physical symptoms may be linked to suicidal behaviors.",
 metadata= {
  "disorder": "Panic Disorder",
  "category": "Anxiety Disorder",
  "section": "suicide_risk",
  "retrieval_priority": "low"
 }
),Document(
 page_content= "## CHUNK 8 — COMORBIDITY: Panic Disorder\n\n* **High Overall Comorbidity:** In the general population, 80% of individuals with panic disorder had a lifetime comorbid mental diagnosis; panic disorder infrequently occurs in clinical settings without other psychopathology.\n* **Other Anxiety Disorders:** Most common comorbid condition, particularly agoraphobia.\n* **Major Depressive Disorder:** Lifetime comorbidity rates vary from 10%–65%; in approximately one-third of cases depression precedes panic disorder; in two-thirds depression occurs coincident with or following panic disorder onset.\n* **Bipolar I and Bipolar II Disorder:** Elevated comorbidity.\n* **Alcohol and Substance Use Disorders:** A subset develops substance use disorder, sometimes representing attempted self-medication of anxiety.\n* **Illness Anxiety Disorder:** Comorbidity is common.\n* **General Medical Conditions:** Significantly comorbid with dizziness, cardiac arrhythmias, hyperthyroidism, asthma, COPD, and irritable bowel syndrome; also elevated rates of mitral valve prolapse and thyroid disease.",
 metadata= {
  "disorder": "Panic Disorder",
  "category": "Anxiety Disorder",
  "section": "comorbidity",
  "chunk_type": "comorbidity",
  "retrieval_priority": "high"
 }
),


Document(
page_content="## CHUNK 1 — OVERVIEW: Reactive Attachment Disorder\n\nReactive attachment disorder is a trauma- and stressor-related disorder characterized by markedly disturbed and developmentally inappropriate attachment behaviors in young children.\n\n**Core Features:**\n* Persistent pattern of **inhibited, emotionally withdrawn behavior toward adult caregivers**.\n* Child **rarely or minimally seeks comfort when distressed**.\n* Child **rarely or minimally responds to comfort when distressed**.\n* Disorder results from **extremes of insufficient care** such as social neglect, repeated caregiver changes, or institutional rearing.\n* Reflects **absent or severely underdeveloped selective attachment** to caregivers.\n* Children show **reduced positive emotions**, limited social responsiveness, and episodes of unexplained irritability, fear, or sadness.\n\n**Developmental Conditions:**\n* Symptoms must appear **before age 5 years**.\n* Child must have a **developmental age of at least 9 months**, when selective attachments are normally formed.\n\n**Specifiers:**\n* **Persistent:** Disorder present for **>12 months**.\n* **Severity – Severe:** All symptoms present and each manifests at high levels.",
metadata={
"disorder": "Reactive Attachment Disorder",
"category": "Trauma- and Stressor-Related Disorders",
"section": "overview",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Reactive Attachment Disorder\n\n**Diagnostic Threshold:**\n\nAll of the following conditions must be met:\n* **Criterion A:** Both symptoms of inhibited attachment behavior must be present.\n* **Criterion B:** At least **2 symptoms** of persistent social and emotional disturbance.\n* **Criterion C:** Evidence of **≥1 pattern of extreme insufficient care**.\n\n---\n\n### Criterion A — Inhibited Attachment Behavior (Both Required)\n1. **Rarely or minimally seeks comfort** from adult caregivers when distressed.\n2. **Rarely or minimally responds to comfort** when distressed.\n\n---\n\n### Criterion B — Persistent Social and Emotional Disturbance (≥2 Required)\n1. **Minimal social and emotional responsiveness** to others.\n2. **Limited positive affect** during interactions.\n3. **Episodes of unexplained irritability, sadness, or fearfulness**, even during nonthreatening interactions with caregivers.\n\n---\n\n### Criterion C — Pattern of Extreme Insufficient Care (≥1 Required)\n1. **Social neglect or deprivation**, with persistent lack of emotional needs for comfort, stimulation, and affection being met.\n2. **Repeated changes of primary caregivers**, limiting opportunities to form stable attachments (e.g., frequent foster care changes).\n3. **Rearing in unusual settings** that severely limit opportunities to form selective attachments (e.g., institutional care with high child-to-caregiver ratios).\n\n---\n\n### Criterion D — Etiological Link\n* The insufficient care described in Criterion C is **presumed responsible** for the disturbed behavior in Criterion A.\n\n---\n\n### Criterion F — Age Requirement\n* Disturbance must be **evident before age 5 years**.\n\n### Criterion G — Developmental Requirement\n* Child must have **developmental age ≥9 months**.",
metadata={
"disorder": "Reactive Attachment Disorder",
"category": "Trauma- and Stressor-Related Disorders",
"section": "symptom_list",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Reactive Attachment Disorder\n\n* DSM diagnostic criteria **do not require a minimum duration threshold** for diagnosis.\n\n**Age Requirements:**\n* Symptoms must be **present before age 5 years**.\n* Child must have **developmental age of at least 9 months**.\n\n**Specifier — Persistent Form:**\n* **Persistent:** Disorder has been present for **more than 12 months**.\n\n**Developmental Course Notes:**\n* Clinical features typically manifest between **9 months and 5 years of age**.\n* In absence of improved caregiving conditions, symptoms **may persist for several years**.",
metadata={
"disorder": "Reactive Attachment Disorder",
"category": "Trauma- and Stressor-Related Disorders",
"section": "duration",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Reactive Attachment Disorder\n\n* The disturbance must cause **clinically significant impairment in functioning**.\n\n**Domains of Impairment:**\n* **Interpersonal functioning:** Severe difficulty forming relationships with caregivers or peers.\n* **Social functioning:** Impaired ability to engage socially with adults or children.\n* **Emotional functioning:** Reduced capacity for positive emotional expression and poor emotion regulation.\n* **Early developmental functioning:** Impairment across multiple domains of early childhood development.\n\n* The disorder significantly affects the child’s **ability to relate interpersonally** and function in social environments.",
metadata={
"disorder": "Reactive Attachment Disorder",
"category": "Trauma- and Stressor-Related Disorders",
"section": "impairment",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Reactive Attachment Disorder\n\n* **Autism Spectrum Disorder Exclusion:**\n  - Criteria for **autism spectrum disorder** must not be met.\n\n* **Developmental Capacity Exclusion:**\n  - Diagnosis should **not be made in children developmentally incapable of forming selective attachments**.\n  - Child must have **developmental age ≥9 months**.\n\n* **Etiological Requirement:**\n  - Disturbance must be **attributable to a pattern of extreme insufficient care** (Criterion C).\n\n* **Substance Exclusion:**\n  - Symptoms must **not be attributable to the physiological effects of substances**.\n\n* **Medical Condition Consideration:**\n  - Symptoms should **not be better explained by another medical condition affecting behavior or development**.\n\n* **Psychotic/Bipolar Disorders:**\n  - Disturbance should **not be better explained by other severe psychiatric conditions when diagnostic criteria are met**.",
metadata={
"disorder": "Reactive Attachment Disorder",
"category": "Trauma- and Stressor-Related Disorders",
"section": "exclusion",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Reactive Attachment Disorder\n\n* **Autism Spectrum Disorder (ASD):**\n  - Both conditions may show reduced positive emotions and social difficulties.\n  - ASD includes **restricted interests, repetitive behaviors, ritualized patterns**, and **specific deficits in social communication**.\n  - Children with ASD typically **form selective attachments**, whereas children with reactive attachment disorder do not.\n  - ASD usually **does not involve a history of severe social neglect**.\n\n* **Intellectual Developmental Disorder (Intellectual Disability):**\n  - Developmentally delayed children demonstrate **social and emotional skills consistent with cognitive level**.\n  - Children with intellectual disability **still form selective attachments once developmental age reaches 7–9 months**.\n  - Reactive attachment disorder shows **marked lack of attachment despite developmental age ≥9 months**.\n\n* **Depressive Disorders in Children:**\n  - Depression may cause **reduced positive affect**, but children typically **seek and respond to comfort from caregivers**.\n  - In reactive attachment disorder, **comfort-seeking and response are markedly reduced or absent**.\n\n* **Adjustment Disorders or Effects of Neglect:**\n  - Emotional distress related to environmental stressors must be differentiated from **persistent attachment disturbance due to severe neglect**.",
metadata={
"disorder": "Reactive Attachment Disorder",
"category": "Trauma- and Stressor-Related Disorders",
"section": "differential",
"retrieval_priority": "medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Reactive Attachment Disorder\n\n* DSM literature **does not provide a defined diagnostic suicide risk criterion** for reactive attachment disorder.\n\n**Associated Emotional Risk Factors:**\n* Persistent **negative emotional states** such as irritability, fearfulness, and sadness.\n* **Impaired emotion regulation** resulting from disrupted attachment development.\n* **Social neglect and early trauma exposure**, which are known risk factors for later mental health problems.\n\n**Clinical Note:**\n* Although suicide risk is **not a core diagnostic feature**, severe early neglect and attachment disturbances may contribute to **later emotional and behavioral difficulties**.",
metadata={
"disorder": "Reactive Attachment Disorder",
"category": "Trauma- and Stressor-Related Disorders",
"section": "suicide_risk",
"retrieval_priority": "low"
}
),Document(
page_content="## CHUNK 8 — COMORBIDITY: Reactive Attachment Disorder\n\n* **Neglect-Associated Developmental Delays:** Cognitive delays, language delays, and stereotypies associated with neglect frequently co-occur with reactive attachment disorder.\n* **Medical Conditions:** Conditions such as severe malnutrition may accompany signs of reactive attachment disorder.\n* **Internalizing Symptoms:** Internalizing symptoms (e.g., sadness, fearfulness, anxiety) may co-occur.\n* **ADHD and Externalizing Behavior Problems:** A relationship between reactive attachment disorder and externalizing behavior problems or ADHD has been suggested but not clearly established.",
metadata={
"disorder": "Reactive Attachment Disorder",
"category": "Trauma- and Stressor-Related Disorders",
"section": "comorbidity",
"chunk_type": "comorbidity",
"retrieval_priority": "high"
}
)
,
Document(
page_content="## CHUNK 1 — OVERVIEW: Disinhibited Social Engagement Disorder\n\nDisinhibited Social Engagement Disorder (DSED) is a trauma- and stressor-related disorder characterized by a persistent pattern in which a child actively approaches and interacts with unfamiliar adults in a culturally inappropriate and overly familiar manner.\n\n**Core Diagnostic Features:**\n* Persistent pattern of **socially disinhibited behavior toward unfamiliar adults**.\n* Child shows **reduced or absent reticence when approaching strangers**.\n* Behavior **violates culturally accepted social boundaries** for the child's age.\n* Disorder occurs in the context of **extreme insufficient caregiving or social neglect**.\n* The neglectful care is presumed to be **causally related to the disinhibited behavior**.\n\n**Developmental Requirement:**\n* Diagnosis requires a **developmental age of ≥ 9 months**, reflecting the developmental stage when selective attachments normally emerge.\n\n**Specifier — Persistent:**\n* Specified if the disorder has been present for **>12 months**.\n\n**Severity Specifier:**\n* **Severe:** Child exhibits **all symptoms of the disorder**, each manifesting at relatively high levels.",
metadata={
"disorder":"Disinhibited Social Engagement Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"overview",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Disinhibited Social Engagement Disorder\n\n**Diagnostic Threshold:**\n* Child must show **≥2 symptoms** of socially disinhibited behavior toward unfamiliar adults.\n\n**Core Behavioral Symptoms:**\n1. **Reduced or absent reticence** in approaching and interacting with unfamiliar adults.\n2. **Overly familiar verbal or physical behavior** that violates culturally sanctioned and age-appropriate social boundaries.\n3. **Diminished or absent checking back** with an adult caregiver after venturing away, even in unfamiliar settings.\n4. **Willingness to go off with an unfamiliar adult** with minimal or no hesitation.\n\n**Additional Diagnostic Conditions:**\n5. **Behavior not limited to impulsivity:** The behaviors must represent **social disinhibition**, not merely impulsivity as seen in disorders such as ADHD.\n6. **History of insufficient care:** Child must have experienced **≥1 form of extreme insufficient caregiving**, including:\n   - Persistent **social neglect or deprivation** (lack of emotional comfort, stimulation, affection).\n   - **Repeated changes of primary caregivers** limiting stable attachments (e.g., multiple foster placements).\n   - **Rearing in unusual institutional settings** with limited opportunities for selective attachments (e.g., institutions with high child-to-caregiver ratios).\n7. **Etiological linkage requirement:** The insufficient care is **presumed responsible** for the disinhibited social behavior.\n\n**Developmental Eligibility:**\n* Child must have **developmental age ≥9 months**.",
metadata={
"disorder":"Disinhibited Social Engagement Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"symptom_list",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Disinhibited Social Engagement Disorder\n\n**Base Diagnostic Duration:**\n* DSM diagnostic criteria **do not require a minimum duration** for the initial diagnosis.\n\n**Specifier — Persistent:**\n* Disorder is specified as **Persistent** if the full diagnostic pattern has been present for **>12 months**.\n\n**Developmental Timing Requirement:**\n* Diagnosis can only be made when the child has a **developmental age of ≥9 months**, reflecting the developmental stage when selective attachments normally emerge.",
metadata={
"disorder":"Disinhibited Social Engagement Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"duration",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Disinhibited Social Engagement Disorder\n\n* The disorder must result in **clinically significant impairment in social functioning**.\n\n**Areas of Functional Impairment:**\n* Impaired ability to **relate appropriately to unfamiliar adults**.\n* Impairment in **general social competence**.\n* Difficulties forming **appropriate peer relationships**.\n* Increased **risk of peer conflicts and victimization**.\n\n* The social disinhibition must represent a **pathological pattern that exceeds culturally accepted norms** for children's behavior toward strangers.",
metadata={
"disorder":"Disinhibited Social Engagement Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"impairment",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Disinhibited Social Engagement Disorder\n\n* **Impulsivity Exclusion:** The behaviors are **not explained solely by impulsivity**, such as that observed in attention-deficit/hyperactivity disorder.\n\n* **Developmental Age Exclusion:** Diagnosis **must not be made if the child has a developmental age <9 months** because selective attachments are not yet developmentally expected.\n\n* **Cultural Norm Exclusion:** The absence of reticence toward strangers must **exceed culturally accepted norms** for children's social behavior.\n\n* **Etiological Requirement:** Diagnosis requires evidence that the behavior occurred **after exposure to extreme insufficient caregiving**.\n\n* **Medical Condition Exclusion:** Symptoms should **not be attributable to another medical condition** affecting social behavior.\n\n* **Substance Exclusion:** Symptoms must **not be attributable to the physiological effects of substances or medications**.",
metadata={
"disorder":"Disinhibited Social Engagement Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"exclusion",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Disinhibited Social Engagement Disorder\n\n* **Attention-Deficit/Hyperactivity Disorder (ADHD):**\n  - ADHD may involve impulsive social behavior.\n  - DSED is distinguished because the child **does not exhibit the core ADHD symptoms of attention deficits or hyperactivity**.\n  - DSED behavior specifically involves **social disinhibition toward unfamiliar adults** rather than generalized impulsivity.\n\n* **Culturally Normative Social Behavior:**\n  - Some cultures may allow more open interaction with unfamiliar adults.\n  - DSED requires behavior that **clearly exceeds culturally expected norms**.\n\n* **Other Conditions Associated With Neglect:**\n  - Developmental delays, language delays, or stereotyped behaviors may occur due to neglect but **do not alone constitute DSED**.\n\n* **Autism Spectrum Disorder (when comorbid):**\n  - Autism may co-occur with DSED but involves **persistent social communication deficits and restricted behaviors**, which differ from the disinhibited approach behavior of DSED.",
metadata={
"disorder":"Disinhibited Social Engagement Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"differential",
"retrieval_priority":"medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Disinhibited Social Engagement Disorder\n\n* **No specific suicide risk criteria are included in DSM diagnostic criteria for Disinhibited Social Engagement Disorder.**\n\n* Current DSM descriptions **do not identify suicidal ideation or suicide attempts as diagnostic features** of this disorder.",
metadata={
"disorder":"Disinhibited Social Engagement Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"suicide_risk",
"retrieval_priority":"low"
}
),Document(
page_content="## CHUNK 8 — COMORBIDITY: Disinhibited Social Engagement Disorder\n\n* **Neglect-Associated Conditions:** Cognitive delays, language delays, and stereotypies associated with neglect may co-occur.\n* **Autism Spectrum Disorder:** May also co-occur with disinhibited social engagement disorder.\n* **ADHD:** In younger children and middle childhood, DSED frequently co-occurs with ADHD; proposed to relate to common impairments in cognitive inhibitory control.\n* **Externalizing Disorders:** Co-occurrence with externalizing disorders is common, particularly in younger children and middle childhood.",
metadata={
"disorder":"Disinhibited Social Engagement Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"comorbidity",
"chunk_type":"comorbidity",
"retrieval_priority":"high"
}
)
,

Document(
page_content="## CHUNK 1 — OVERVIEW: Posttraumatic Stress Disorder (PTSD)\n\nPosttraumatic Stress Disorder (PTSD) is a trauma- and stressor-related disorder that develops following exposure to **actual or threatened death, serious injury, or sexual violence**.\n\n**Core Diagnostic Domains:**\n- Trauma exposure\n- Intrusion symptoms\n- Avoidance of trauma-related stimuli\n- Negative alterations in cognition and mood\n- Alterations in arousal and reactivity\n\n**Diagnostic Structure:**\n- Trauma exposure (Criterion A)\n- ≥1 intrusion symptom (Criterion B)\n- ≥1 avoidance symptom (Criterion C)\n- ≥2 negative cognition/mood symptoms (Criterion D)\n- ≥2 arousal/reactivity symptoms (Criterion E)\n\n**Specifiers:**\n- With dissociative symptoms (depersonalization or derealization)\n- With delayed expression (full criteria met ≥6 months after trauma)",
metadata={
"disorder":"Posttraumatic Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"overview",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Posttraumatic Stress Disorder\n\n**Step 1 — Trauma Exposure Requirement (Criterion A):**\nExposure to **actual or threatened death, serious injury, or sexual violence** via ≥1 of the following:\n1. Directly experiencing the traumatic event.\n2. Witnessing the event in person.\n3. Learning the event occurred to a close family member or close friend (must be violent or accidental death).\n4. Repeated or extreme exposure to aversive details of traumatic events (e.g., first responders).\n\n**Step 2 — Intrusion Symptoms (Criterion B)**\n**Diagnostic Threshold:** ≥1 symptom\n\n1. Recurrent, involuntary, intrusive distressing memories.\n2. Recurrent distressing dreams related to the trauma.\n3. Dissociative reactions (e.g., flashbacks).\n4. Intense psychological distress at trauma reminders.\n5. Marked physiological reactions to trauma reminders.\n\n**Step 3 — Avoidance Symptoms (Criterion C)**\n**Diagnostic Threshold:** ≥1 symptom\n\n1. Avoidance of distressing trauma-related memories, thoughts, or feelings.\n2. Avoidance of external reminders (people, places, activities, conversations).\n\n**Step 4 — Negative Alterations in Cognition and Mood (Criterion D)**\n**Diagnostic Threshold:** ≥2 symptoms\n\n1. Inability to remember important aspects of trauma.\n2. Persistent exaggerated negative beliefs about self, others, or world.\n3. Distorted blame of self or others about trauma cause/consequences.\n4. Persistent negative emotional state (fear, guilt, anger, shame).\n5. Markedly diminished interest in activities.\n6. Feelings of detachment or estrangement.\n7. Persistent inability to experience positive emotions.\n\n**Step 5 — Alterations in Arousal and Reactivity (Criterion E)**\n**Diagnostic Threshold:** ≥2 symptoms\n\n1. Irritable behavior or angry outbursts.\n2. Reckless or self-destructive behavior.\n3. Hypervigilance.\n4. Exaggerated startle response.\n5. Problems with concentration.\n6. Sleep disturbance.",
metadata={
"disorder":"Posttraumatic Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"symptom_list",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Posttraumatic Stress Disorder\n\n**Minimum Duration Requirement:**\n- Symptoms from Criteria **B, C, D, and E must persist for >1 month**.\n\n**Delayed Expression Specifier:**\n- Full diagnostic criteria are **not met until ≥6 months after the traumatic event**.\n\n**Onset:**\n- Some symptoms may begin **immediately after the trauma**, but full syndrome may emerge later.",
metadata={
"disorder":"Posttraumatic Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"duration",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Posttraumatic Stress Disorder\n\nThe disturbance must cause **clinically significant distress or impairment** in at least one domain:\n\n- Social functioning\n- Occupational functioning\n- Other important areas of functioning\n\nSymptoms must represent **clinically meaningful dysfunction following trauma exposure**.",
metadata={
"disorder":"Posttraumatic Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"impairment",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Posttraumatic Stress Disorder\n\nDiagnosis requires exclusion of alternative causes.\n\n**Exclusions:**\n\n- **Substance exclusion:**\n  - Symptoms must not be attributable to physiological effects of **substances (e.g., alcohol, medication)**.\n\n- **Medical condition exclusion:**\n  - Symptoms must not be explained by **another medical condition**.\n\n- **Dissociative subtype rule:**\n  - Depersonalization or derealization must not be attributable to **substances or neurological conditions** (e.g., seizures).",
metadata={
"disorder":"Posttraumatic Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"exclusion",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Posttraumatic Stress Disorder\n\n**Acute Stress Disorder:**\n- Similar symptom profile but **duration is 3 days to 1 month**, whereas PTSD requires **>1 month**.\n\n**Adjustment Disorder:**\n- Emotional/behavioral symptoms occur after stressor but **do not meet full PTSD trauma or symptom criteria**.\n\n**Anxiety Disorder:**\n- Anxiety symptoms may overlap but **lack specific trauma exposure and re-experiencing symptoms**.\n\n**Obsessive-Compulsive Disorder:**\n- Intrusive thoughts occur but **are not trauma-related memories**.\n\n**Major Depressive Disorder:**\n- May involve anhedonia and negative mood but **lacks trauma-linked intrusion and avoidance clusters**.",
metadata={
"disorder":"Posttraumatic Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"differential",
"retrieval_priority":"medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Posttraumatic Stress Disorder\n\n**Suicide Risk Indicators:**\n- Reckless or self-destructive behavior (Criterion E symptom).\n- Persistent negative emotional states.\n- Feelings of detachment and hopelessness.\n\n**Clinical Implication:**\n- Individuals with PTSD may have **elevated risk of suicidal ideation and suicide attempts**, particularly when comorbid depression, substance use, or severe trauma exposure is present.",
metadata={
"disorder":"Posttraumatic Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"suicide_risk",
"retrieval_priority":"low"
}
),Document(
page_content="## CHUNK 8 — COMORBIDITY: Posttraumatic Stress Disorder\n\n* **High Overall Comorbidity:** Individuals with PTSD are more likely than those without PTSD to have symptoms meeting criteria for at least one other mental disorder.\n* **Depressive Disorders:** Frequently comorbid, including major depressive disorder.\n* **Bipolar Disorders:** Co-occurrence with bipolar disorders is elevated.\n* **Anxiety Disorders:** Commonly comorbid anxiety disorders.\n* **Substance Use Disorders:** Elevated co-occurrence; may represent self-medication.\n* **Major Neurocognitive Disorder:** PTSD is associated with increased risk of major neurocognitive disorder.\n* **Traumatic Brain Injury (TBI):** Women are more likely to develop PTSD following mild TBI.\n* **Children — Different Comorbidity Pattern:** Although most young children with PTSD also have at least one other diagnosis, the comorbidity pattern differs from adults, with oppositional defiant disorder and separation anxiety disorder predominating in children.",
metadata={
"disorder":"Posttraumatic Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"comorbidity",
"chunk_type":"comorbidity",
"retrieval_priority":"high"
}
)
,

Document(
page_content="## CHUNK 1 — OVERVIEW: Prolonged Grief Disorder\n\n**Category:** Trauma- and Stressor-Related Disorders\n\nProlonged Grief Disorder is a persistent and maladaptive grief response that occurs after the death of a person who was close to the bereaved individual. The condition is characterized by intense longing or preoccupation with the deceased, accompanied by additional grief-related symptoms that persist beyond culturally expected mourning periods.\n\n**Core Features:**\n* Death of a close person followed by persistent grief reaction.\n* Persistent yearning/longing or preoccupation with the deceased.\n* Additional cognitive, emotional, or behavioral symptoms related to the loss.\n* Symptoms exceed expected **social, cultural, or religious mourning norms**.\n\n**Minimum Diagnostic Conditions:**\n* Death of a close individual must have occurred **≥12 months ago in adults**.\n* For **children and adolescents**, death must have occurred **≥6 months ago**.\n* Presence of core grief symptoms plus additional associated symptoms.\n* Symptoms must produce clinically significant distress or impairment.",
metadata={
"disorder":"Prolonged Grief Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"overview",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Prolonged Grief Disorder\n\n**Diagnostic Threshold:**\n* **≥1 core symptom from Criterion B** AND\n* **≥3 additional symptoms from Criterion C**.\n* Symptoms must occur **most days to a clinically significant degree**.\n* Symptoms must occur **nearly every day for at least the last 1 month**.\n\n**Criterion B — Core Grief Symptoms (At least 1 required):**\n1. **Intense yearning or longing for the deceased person.**\n2. **Preoccupation with thoughts or memories of the deceased person.**\n   * In children/adolescents, preoccupation may focus on **circumstances of the death**.\n\n**Criterion C — Additional Symptoms (≥3 required):**\n1. **Identity disruption** (e.g., feeling as though part of oneself has died).\n2. **Marked disbelief** about the death.\n3. **Avoidance of reminders** that the person is dead.\n4. **Intense emotional pain** related to the death (e.g., anger, bitterness, sorrow).\n5. **Difficulty reintegrating into relationships or activities** after the death (e.g., problems engaging with friends, interests, or future planning).\n6. **Emotional numbness** or marked reduction in emotional experience.\n7. **Feeling that life is meaningless** because of the death.\n8. **Intense loneliness** resulting from the loss.",
metadata={
"disorder":"Prolonged Grief Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"symptom_list",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Prolonged Grief Disorder\n\n**Minimum Time Since Death:**\n* **Adults:** Death of the close person must have occurred **≥12 months ago**.\n* **Children and Adolescents:** Death must have occurred **≥6 months ago**.\n\n**Symptom Duration Requirement:**\n* Core grief symptoms and additional symptoms must occur **most days** to a clinically significant degree.\n* Symptoms must occur **nearly every day for at least the last 1 month**.\n\n**Persistence Requirement:**\n* The grief reaction must persist **beyond culturally expected mourning periods**.\n* The severity and duration must clearly **exceed expected social, cultural, or religious norms** for the individual's context.",
metadata={
"disorder":"Prolonged Grief Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"duration",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Prolonged Grief Disorder\n\n* Symptoms must cause **clinically significant distress** OR **impairment** in functioning.\n\n**Areas of Functional Impairment May Include:**\n* Social functioning\n* Occupational functioning\n* Educational functioning\n* Other important life domains\n\n* The disturbance must significantly interfere with the individual's ability to engage in relationships, daily activities, or future life planning.",
metadata={
"disorder":"Prolonged Grief Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"impairment",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Prolonged Grief Disorder\n\n* **Cultural Norm Exclusion:** Symptoms must clearly **exceed expected social, cultural, or religious norms** for bereavement in the individual's culture.\n\n* **Mental Disorder Exclusion:** Symptoms are **not better explained by another mental disorder**, including:\n  - Major Depressive Disorder\n  - Posttraumatic Stress Disorder\n\n* **Substance Exclusion:** Symptoms are **not attributable to the physiological effects of a substance** (e.g., medication, alcohol, drugs).\n\n* **Medical Condition Exclusion:** Symptoms are **not attributable to another medical condition**.",
metadata={
"disorder":"Prolonged Grief Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"exclusion",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Prolonged Grief Disorder\n\n* **Normal Grief:** Normal bereavement may involve intense sadness and longing, but symptoms typically decrease over time and remain within expected cultural norms. Prolonged grief disorder requires **persistent severe grief ≥12 months (≥6 months in children/adolescents)** with functional impairment.\n\n* **Major Depressive Disorder (MDD):** Both conditions may involve sadness, crying, and suicidal thoughts. In prolonged grief disorder, distress is primarily **focused on separation from the deceased**, whereas MDD involves **pervasive depressed mood or loss of interest across contexts**.\n\n* **Persistent Depressive Disorder:** Chronic depressive mood without the specific grief-focused longing or preoccupation with the deceased.\n\n* **Posttraumatic Stress Disorder (PTSD):** Both disorders may include intrusive thoughts and avoidance. In PTSD, intrusions focus on the **traumatic event**, whereas in prolonged grief disorder intrusions focus on **memories or longing for the deceased**.\n\n* **Separation Anxiety Disorder:** Involves fear or anxiety regarding **separation from living attachment figures**, whereas prolonged grief disorder involves separation from a **deceased person**.\n\n* **Psychotic Disorders:** Hallucinations or perceptions of the deceased may occur in normal grief and prolonged grief disorder. Diagnosis of a psychotic disorder requires **additional psychotic symptoms** such as delusions, disorganized thinking, or negative symptoms.",
metadata={
"disorder":"Prolonged Grief Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"differential",
"retrieval_priority":"medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Prolonged Grief Disorder\n\n* Individuals with prolonged grief disorder symptoms have an **elevated risk of suicidal ideation**.\n\n**Risk Characteristics:**\n* Suicidal ideation may occur **even after controlling for Major Depressive Disorder and PTSD**.\n* Risk is observed **across the lifespan and across countries**.\n\n**Factors Associated With Increased Suicide Risk:**\n* Social isolation and perceived lack of belonging.\n* Persistent psychological distress and avoidance behaviors.\n* Stigma related to bereavement.\n* Bereavement following **violent deaths** (e.g., homicide, suicide, accidents).\n* Death of a **child**, particularly when the child was **younger than 25 years**.\n\n* Current evidence shows increased **suicidal ideation**, although the relationship with increased rates of suicide attempts or completed suicide remains unclear.",
metadata={
"disorder":"Prolonged Grief Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"suicide_risk",
"retrieval_priority":"low"
}
),

Document(
page_content="## CHUNK 1 — OVERVIEW: Acute Stress Disorder\n\n**Category:** Trauma- and Stressor-Related Disorders\n\nAcute Stress Disorder is characterized by the development of anxiety, intrusion, dissociation, avoidance, and arousal symptoms following exposure to a traumatic event.\n\n**Core Features:**\n* Exposure to a traumatic event involving actual or threatened death, serious injury, or sexual violence.\n* Development of multiple trauma-related psychological symptoms.\n* Symptoms begin or worsen after the traumatic exposure.\n* Symptoms last for a short-term period following trauma exposure.\n\n**Criterion A — Trauma Exposure (at least one required):**\n1. **Directly experiencing** the traumatic event(s).\n2. **Witnessing, in person**, the event(s) as it occurred to others.\n3. **Learning that the traumatic event occurred to a close family member or close friend.**\n   * If the event involved death or threatened death, it must have been **violent or accidental**.\n4. **Repeated or extreme exposure to aversive details of traumatic events** (e.g., first responders collecting human remains, police repeatedly exposed to details of child abuse).\n   * Exposure through electronic media, television, movies, or pictures **does not qualify** unless it is **work-related**.",
metadata={
"disorder":"Acute Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"overview",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Acute Stress Disorder\n\n**Diagnostic Threshold:**\n* **≥9 symptoms** must be present.\n* Symptoms must come from **any of the five categories:** intrusion, negative mood, dissociation, avoidance, and arousal.\n* Symptoms must **begin or worsen after the traumatic event**.\n\n**Intrusion Symptoms:**\n1. **Recurrent, involuntary, and intrusive distressing memories** of the traumatic event(s).\n   * In children: repetitive play may express trauma themes.\n2. **Recurrent distressing dreams** related to the traumatic event(s).\n   * In children: frightening dreams without recognizable content may occur.\n3. **Dissociative reactions (flashbacks)** in which the individual feels or acts as if the traumatic event is recurring.\n   * May include complete loss of awareness of present surroundings.\n   * In children: trauma reenactment may occur in play.\n4. **Intense or prolonged psychological distress** or **marked physiological reactions** when exposed to internal or external cues resembling the trauma.\n\n**Negative Mood:**\n5. **Persistent inability to experience positive emotions** (e.g., inability to feel happiness, satisfaction, or loving feelings).\n\n**Dissociative Symptoms:**\n6. **Altered sense of reality of surroundings or oneself** (e.g., depersonalization, derealization, being in a daze, time slowing).\n7. **Inability to remember an important aspect of the traumatic event** due to dissociative amnesia (not due to head injury, alcohol, or drugs).\n\n**Avoidance Symptoms:**\n8. **Efforts to avoid distressing memories, thoughts, or feelings** about or closely associated with the traumatic event.\n9. **Efforts to avoid external reminders** (people, places, conversations, activities, objects, situations) that arouse trauma-related memories or feelings.\n\n**Arousal Symptoms:**\n10. **Sleep disturbance** (difficulty falling asleep, difficulty staying asleep, or restless sleep).\n11. **Irritable behavior and angry outbursts**, often expressed as verbal or physical aggression toward people or objects.\n12. **Hypervigilance**.\n13. **Problems with concentration**.\n14. **Exaggerated startle response**.",
metadata={
"disorder":"Acute Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"symptom_list",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Acute Stress Disorder\n\n* **Minimum duration:** Symptoms must persist for **at least 3 days** after the traumatic event.\n* **Maximum duration:** Symptoms must resolve within **1 month after trauma exposure**.\n* **Diagnostic window:** The disorder can only be diagnosed **between 3 days and 1 month following trauma exposure**.\n* Symptoms that occur immediately after the trauma but **resolve in less than 3 days do not meet criteria** for Acute Stress Disorder.\n* If symptoms **persist beyond 1 month**, the diagnosis should be reconsidered (e.g., possible Posttraumatic Stress Disorder).",
metadata={
"disorder":"Acute Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"duration",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Acute Stress Disorder\n\n* The disturbance must cause **clinically significant distress** or **impairment** in functioning.\n\n**Functional areas that may be impaired include:**\n* Social functioning\n* Occupational functioning\n* Interpersonal functioning\n* Other important areas of daily life\n\n* Symptoms may interfere with sleep, concentration, energy levels, or the ability to attend to daily responsibilities.",
metadata={
"disorder":"Acute Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"impairment",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Acute Stress Disorder\n\n* **Substance Exclusion:** The disturbance is **not attributable to the physiological effects of a substance** (e.g., medication, alcohol, drugs).\n\n* **Medical Condition Exclusion:** The disturbance is **not attributable to another medical condition** (e.g., mild traumatic brain injury).\n\n* **Psychotic Disorder Exclusion:** The disturbance is **not better explained by brief psychotic disorder**.\n\n* **Symptom Attribution Rule:** Symptoms must be attributable to trauma exposure rather than other neurological or physiological causes.",
metadata={
"disorder":"Acute Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"exclusion",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Acute Stress Disorder\n\n* **Adjustment Disorders:** Stressor may be less severe than trauma required for Acute Stress Disorder. Diagnosis used when symptoms occur after stressors that do not meet trauma exposure criteria or when ASD symptom threshold is not met.\n\n* **Panic Disorder:** Panic attacks can occur in Acute Stress Disorder, but Panic Disorder requires **unexpected recurrent panic attacks** and **persistent concern about future attacks**.\n\n* **Dissociative Disorders:** Severe dissociative symptoms without the broader trauma symptom pattern may indicate **depersonalization/derealization disorder** or **dissociative amnesia**.\n\n* **Posttraumatic Stress Disorder (PTSD):** Distinguished primarily by **duration**. PTSD is diagnosed when trauma-related symptoms **persist for more than 1 month**.\n\n* **Obsessive-Compulsive Disorder:** Intrusive thoughts in OCD are **obsessions unrelated to trauma**, and compulsions are typically present.\n\n* **Psychotic Disorders:** Flashbacks in ASD must be distinguished from hallucinations or delusions seen in psychotic disorders.\n\n* **Traumatic Brain Injury (TBI):** Neurocognitive symptoms following head injury may overlap with ASD symptoms, but **reexperiencing and avoidance are characteristic of ASD**, whereas **persistent confusion and disorientation are more typical of TBI-related neurocognitive disorders**.",
metadata={
"disorder":"Acute Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"differential",
"retrieval_priority":"medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Acute Stress Disorder\n\n* Acute Stress Disorder may be associated with **severe psychological distress**, anxiety, and emotional dysregulation following trauma exposure.\n\n**Potential Risk Factors:**\n* Intense emotional reactions to trauma.\n* Catastrophic thinking about the traumatic event or future harm.\n* Severe anxiety and panic reactions.\n* Impulsive or chaotic behavior following trauma.\n\n* Although **suicidal ideation is not a diagnostic criterion**, the level of acute distress following trauma may increase vulnerability to suicidal thoughts in some individuals.",
metadata={
"disorder":"Acute Stress Disorder",
"category":"Trauma- and Stressor-Related Disorders",
"section":"suicide_risk",
"retrieval_priority":"low"
}
)

,
 Document( page_content= "## CHUNK 1 — OVERVIEW: Obsessive-Compulsive Disorder (OCD)\n\nObsessive-Compulsive Disorder (OCD) is characterized by the presence of obsessions, compulsions, or both that are time-consuming or cause clinically significant distress or impairment. Obsessions are intrusive, unwanted thoughts, urges, or images causing marked anxiety, while compulsions are repetitive behaviors or mental acts performed to neutralize or reduce distress, often excessive or unrealistic.\n\nCore Features:\n* Obsessions and/or compulsions.\n* Compulsions are often thematically linked to obsessions (e.g., contamination and washing; harm and checking).\n* Behaviors are not performed for pleasure, but for temporary relief from distress.\n* Symptoms can occur in multiple dimensions: cleaning, symmetry, forbidden/taboo thoughts, and harm.\n* Insight varies: good/fair, poor, or absent/delusional.\n* Tic-related specifier if there is a current or past history of tic disorder.", metadata= { "disorder": "Obsessive-Compulsive Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "overview", "retrieval_priority": "high" } ),
 Document( page_content= "## CHUNK 2 — SYMPTOM LIST: Obsessive-Compulsive Disorder\n\nDiagnostic Threshold:\n* Presence of obsessions, compulsions, or both.\n* Obsessions or compulsions must be time-consuming (≥1 hour per day) or cause clinically significant distress or impairment.\n\nObsessions:\n1. Recurrent and persistent thoughts, urges, or images that are intrusive and unwanted.\n2. Cause marked anxiety or distress in most individuals.\n3. Attempts to ignore, suppress, or neutralize the obsessions (e.g., with a compulsion).\n\nCompulsions:\n1. Repetitive behaviors (e.g., hand washing, ordering, checking) or mental acts (e.g., praying, counting, repeating words silently).\n2. Driven to perform in response to an obsession or according to rigid rules.\n3. Aimed at preventing/reducing anxiety or preventing a dreaded event; behavior is excessive or not realistically connected to the feared event.\n\nAdditional Features:\n* Obsessions and compulsions may vary across dimensions: cleaning, symmetry, forbidden/taboo thoughts, harm.\n* Sensory phenomena (e.g., 'just-right' feelings) may precede compulsions.\n* Dysfunctional beliefs (inflated responsibility, overestimation of threat, need to control thoughts) may be present.", metadata= { "disorder": "Obsessive-Compulsive Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "symptom_list", "retrieval_priority": "high" } ),
   Document( page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Obsessive-Compulsive Disorder\n\n* Time Requirement: Obsessions and/or compulsions must be time-consuming (≥1 hour per day) or cause clinically significant distress or impairment.\n* Course: Symptoms are often chronic with waxing and waning; onset is typically gradual.\n* Childhood Onset: Early-onset OCD may persist into adulthood if untreated; 40% may experience remission by early adulthood.", metadata= { "disorder": "Obsessive-Compulsive Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "duration", "retrieval_priority": "high" } ),
     Document( page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENT: Obsessive-Compulsive Disorder\n\n* Obsessions and compulsions must cause clinically significant distress or impairment in social, occupational, or other important areas of functioning.\n* Functional consequences may include:\n - Avoidance of triggering situations.\n - Interference with relationships and family functioning.\n - Impaired school or work performance.\n - Developmental difficulties in children/adolescents.\n - Health consequences (e.g., dermatological issues from excessive washing).\n* Distress or impairment can result from the time spent on compulsions, avoidance behaviors, or the specific nature of obsessions/compulsions.", metadata= { "disorder": "Obsessive-Compulsive Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "impairment", "retrieval_priority": "high" } ),
       Document( page_content= "## CHUNK 5 — EXCLUSION CRITERIA: Obsessive-Compulsive Disorder\n\n* Substance Exclusion: Symptoms are not attributable to the physiological effects of a substance (drug of abuse or medication).\n* Medical Condition Exclusion: Symptoms are not attributable to another medical condition.\n* Other Mental Disorder Exclusion: \n - Excessive worries (GAD), preoccupation with appearance (BDD), difficulty discarding items (hoarding), hair pulling (trichotillomania), skin picking (excoriation), ritualized eating behaviors (eating disorders), preoccupation with substances/gambling (addictive disorders), illness anxiety, sexual urges/fantasies (paraphilic disorders), guilty ruminations (MDD), thought insertion/delusions (psychotic disorders), or repetitive behaviors (autism spectrum disorder) do not explain the symptoms.\n* Psychotic Disorder Exclusion: OCD with absent insight/delusional beliefs is not diagnosed as a psychotic disorder if obsessions/compulsions are present.", metadata= { "disorder": "Obsessive-Compulsive Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "exclusion", "retrieval_priority": "high" } ),
         Document( page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Obsessive-Compulsive Disorder\n\n* Anxiety Disorder: GAD involves worries about real-life concerns; OCD obsessions are intrusive, irrational, and often accompanied by compulsions. Specific phobia fears are circumscribed without rituals. Social anxiety focuses on social interactions/performance.\n* Major Depressive Disorder: Rumination is mood-congruent, not linked to compulsions.\n* Other Obsessive-Compulsive and Related Disorders: BDD (appearance), trichotillomania (hair pulling), hoarding disorder (difficulty discarding items) without typical OCD obsessions.\n* Eating Disorders: OCD is not limited to weight/food concerns.\n* Tics and Stereotyped Movements: Tics/stereotypies are nonfunctional, less complex, and not linked to obsessions.\n* Psychotic Disorders: Delusional OCD beliefs exist without other psychotic features.\n* Other Compulsive-like Behaviors: Sexual behaviors, gambling, or substance use differ as they are usually pleasurable.\n* Obsessive-Compulsive Personality Disorder: Involves pervasive maladaptive perfectionism and control; not characterized by intrusive thoughts or compulsions.", metadata= { "disorder": "Obsessive-Compulsive Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "differential", "retrieval_priority": "medium" } ),
           Document( page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Obsessive-Compulsive Disorder\n\n* Suicidal Ideation: Lifetime suicidal ideation reported in up to 44.1% of individuals; current suicidal ideation in ~25.9%.\n* Suicide Attempts: Lifetime attempts in ~14.2%; current suicidal thoughts, plans, and attempts vary across populations.\n* Risk Predictors: Greater OCD severity, unacceptable thoughts dimension, comorbid depressive/anxiety disorders, past history of suicidality.\n* Comorbidity: Substance use disorders and personality disorders increase risk; female gender, higher parental education, and comorbid anxiety disorders are protective.\n* Clinical Assessment: Suicide risk should be evaluated in all individuals diagnosed with OCD.", metadata= { "disorder": "Obsessive-Compulsive Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "suicide_risk", "retrieval_priority": "low" } ) 

,
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Obsessive-Compulsive Disorder\n\n* **Anxiety Disorders (76%):** Panic disorder, social anxiety disorder, generalized anxiety disorder, and specific phobia are frequently comorbid.\n* **Depressive and Bipolar Disorders (63%):** Major depressive disorder is the most common comorbid condition (41%); bipolar disorder is also elevated compared to the general population.\n* **Impulse-Control Disorders (56%):** A substantial proportion of individuals with OCD have comorbid impulse-control disorders.\n* **Substance Use Disorder (39%):** Co-occurrence with alcohol and other substance use disorders is common.\n* **Tic Disorders (up to 30%):** Most prevalent in males with childhood-onset OCD; a triad of OCD, tic disorder, and ADHD is frequently observed in children.\n* **OC-Related Disorders:** Body dysmorphic disorder, trichotillomania, and excoriation disorder occur at elevated rates in individuals with OCD.\n* **Schizophrenia/Schizoaffective Disorder:** OCD prevalence is approximately 12% in individuals with schizophrenia or schizoaffective disorder.\n* **Eating Disorders:** Elevated rates of anorexia nervosa, bulimia nervosa, and body dysmorphic disorder reported.\n* **Tourette's Disorder:** Comorbidity is well-established, particularly in tic-related OCD.\n* **OC Personality Disorder:** Present in 23–32% of treatment-seeking adults with OCD.",
    metadata={
        "disorder": "Obsessive-Compulsive Disorder",
        "category": "Obsessive-Compulsive and Related Disorders",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
),
Document(
page_content="## CHUNK 1 — OVERVIEW: Body Dysmorphic Disorder (BDD)\n\nBody Dysmorphic Disorder is characterized by a persistent **preoccupation with perceived defects or flaws in physical appearance** that are **not observable or appear slight to others**.\n\n**Core Diagnostic Characteristics:**\n\n- Primary feature: Preoccupation with one or more perceived defects in appearance.\n- The perceived defects are **minimal or not observable to other people**.\n- Concerns may involve **one or multiple body areas**.\n\n**Common Body Areas of Concern:**\n1. Skin (e.g., acne, scars, wrinkles, paleness).\n2. Hair (e.g., thinning hair, excessive body or facial hair).\n3. Nose (e.g., size or shape).\n4. Face shape or facial symmetry.\n5. Eyes, lips, teeth, chin, eyebrows.\n6. Breasts, chest, stomach, legs, or genitals.\n\n**Nature of Preoccupation:**\n- Intrusive and unwanted thoughts about appearance.\n- Difficult to control or resist.\n- **Time-consuming**, typically **3–8 hours per day**.\n\n**Associated Behavioral Pattern:**\n- Individuals perform repetitive behaviors or mental acts in response to appearance concerns.\n\n**Common Clinical Features:**\n- Persistent dissatisfaction with appearance.\n- Attempts to hide or camouflage perceived defects.\n- Frequent comparison of appearance with others.\n- High levels of shame and distress related to perceived physical flaws.",
metadata={
"disorder":"Body Dysmorphic Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"overview",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Body Dysmorphic Disorder\n\n**Diagnostic Threshold:**\n- **All of the following core criteria must be present.**\n\n**Symptoms:**\n\n1. **Preoccupation with perceived defects or flaws in physical appearance** that are **not observable or appear slight to others**.\n\n2. **Repetitive behaviors or mental acts performed in response to appearance concerns.**\n\n**Examples of Repetitive Behaviors:**\n- Mirror checking.\n- Excessive grooming (e.g., combing, styling, shaving, plucking hair).\n- Skin picking intended to improve perceived defects.\n- Touching or examining disliked body areas.\n- Taking excessive photographs or selfies.\n- Excessive tanning.\n- Repeatedly changing clothing to hide perceived flaws.\n- Compulsive shopping for beauty products.\n- Seeking reassurance from others.\n\n**Examples of Mental Acts:**\n- Repeatedly comparing one's appearance with others.\n- Mentally evaluating perceived defects.\n\n**Behavioral Characteristics:**\n- Behaviors are **driven and difficult to resist**.\n- Behaviors are **not pleasurable**.\n- Behaviors may **increase anxiety or dysphoria**.\n- Behaviors are typically **time-consuming**.",
metadata={
"disorder":"Body Dysmorphic Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"symptom_list",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Body Dysmorphic Disorder\n\n- DSM diagnostic criteria **do not specify a minimum duration requirement**.\n- However, the preoccupation and associated behaviors are typically:\n  - **Persistent and chronic**.\n  - **Time-consuming**, often occupying **3–8 hours per day**.\n\n**Course Characteristics:**\n- Mean age of onset: **16–17 years**.\n- Median age of onset: **15 years**.\n- Most common age of onset: **12–13 years**.\n- **Approximately two-thirds of cases begin before age 18**.\n\n**Typical Illness Course:**\n- Symptoms often begin with **subclinical appearance concerns in early adolescence**.\n- Concerns may **gradually evolve into full diagnostic disorder**.\n- The disorder is **usually chronic** without treatment.\n- Improvement is possible with **evidence-based treatment**.",
metadata={
"disorder":"Body Dysmorphic Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"duration",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Body Dysmorphic Disorder\n\n- The appearance preoccupation must cause **clinically significant distress** or **impairment** in functioning.\n\n**Areas of Functional Impairment:**\n1. Social functioning.\n2. Occupational functioning.\n3. Academic functioning.\n4. Other important life roles.\n\n**Examples of Functional Consequences:**\n- Avoidance of social situations.\n- Avoidance of work or school.\n- Poor job or academic performance.\n- Missing work or school due to appearance concerns.\n- Being housebound in severe cases.\n\n**Severity Range:**\n- Impairment may range from **moderate social avoidance** to **severe incapacitation**.\n\n**Additional Functional Impact:**\n- Markedly reduced quality of life.\n- Significant psychosocial impairment.\n- Approximately **20% of youths report dropping out of school primarily due to BDD symptoms**.",
metadata={
"disorder":"Body Dysmorphic Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"impairment",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Body Dysmorphic Disorder\n\n**Body Weight/Eating Disorder Exclusion:**\n- The appearance preoccupation **must not be better explained by concerns with body fat or weight in an individual whose symptoms meet diagnostic criteria for an eating disorder**.\n\n**Substance Exclusion:**\n- Symptoms must **not be attributable to the physiological effects of a substance** (e.g., drugs of abuse, medications).\n\n**Medical Condition Exclusion:**\n- Symptoms must **not be attributable to another medical condition**.\n\n**Psychotic Disorder Clarification:**\n- Delusional appearance beliefs **are diagnosed as Body Dysmorphic Disorder with absent insight/delusional beliefs**, rather than as a primary psychotic disorder.\n\n**Gender Dysphoria Exclusion:**\n- Do not diagnose BDD if the preoccupation is limited to discomfort with **primary or secondary sex characteristics** in individuals with **gender dysphoria**.\n\n**Other Condition Exclusions:**\n- Do not diagnose BDD when the concern is primarily about **believing one emits a foul body odor** (olfactory reference disorder).\n- Do not diagnose BDD when the concern is about **body configuration mismatch with desire for amputation** (body integrity dysphoria).\n- Do not diagnose BDD when appearance concerns involve **clearly observable physical defects** that are not slight.",
metadata={
"disorder":"Body Dysmorphic Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"exclusion",
"retrieval_priority":"high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Body Dysmorphic Disorder\n\n**1. Normal Appearance Concerns**\n- Normal concerns about appearance are **not excessive, time-consuming, or impairing**.\n- BDD involves **intrusive preoccupations and repetitive behaviors** causing distress or impairment.\n\n**2. Eating Disorders**\n- Eating disorders involve **preoccupation with body weight or fat**.\n- In BDD, appearance concerns focus on **specific body features rather than weight**.\n\n**3. Obsessive-Compulsive Disorder (OCD)**\n- OCD obsessions are **not limited to appearance concerns**.\n- BDD obsessions and compulsions are **specifically focused on physical appearance**.\n\n**4. Excoriation (Skin-Picking) Disorder**\n- Diagnosed when skin picking occurs **without appearance preoccupation**.\n- If skin picking is performed **to improve perceived skin defects**, diagnose **BDD**.\n\n**5. Trichotillomania (Hair-Pulling Disorder)**\n- Hair pulling occurs due to urges unrelated to appearance concerns.\n- If hair removal occurs **to improve perceived appearance defects**, diagnose **BDD**.\n\n**6. Major Depressive Disorder**\n- Depression may involve low self-esteem and body dissatisfaction.\n- BDD is distinguished by **prominent appearance preoccupation and repetitive behaviors**.\n\n**7. Anxiety Disorder**\n- Social anxiety disorder involves **fear of social evaluation**.\n- In BDD, social anxiety is **secondary to perceived appearance defects**.\n\n**8. Psychotic Disorders**\n- Delusional appearance beliefs may occur in BDD.\n- BDD is diagnosed when **appearance preoccupation and repetitive behaviors are present without broader psychotic symptoms**.",
metadata={
"disorder":"Body Dysmorphic Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"differential",
"retrieval_priority":"medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Body Dysmorphic Disorder\n\n**Elevated Suicide Risk:**\n- Individuals with Body Dysmorphic Disorder have **significantly increased rates of suicidal thoughts and behaviors**.\n\n**Relative Risk (Research Findings):**\n- Individuals with BDD are approximately **4 times more likely to experience suicidal thoughts** compared with control groups.\n- Individuals with BDD are approximately **2.6 times more likely to attempt suicide** compared with control groups.\n\n**Observed Prevalence (General Population Studies):**\n- Suicidal thoughts in BDD: **19%–31%**.\n- Suicidal behaviors in BDD: **7%–22%**.\n\n**Risk Factors Associated With Suicide in BDD:**\n1. Severity of body dysmorphic symptoms.\n2. High levels of shame and appearance-related distress.\n3. Low self-esteem.\n4. Comorbid psychiatric disorders.\n5. Perceived abuse or trauma.\n6. Unemployment or social isolation.\n\n**Clinical Attribution:**\n- Many individuals with BDD report that **suicidal thoughts or suicide attempts are directly related to their appearance concerns**.",
metadata={
"disorder":"Body Dysmorphic Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"suicide_risk",
"retrieval_priority":"low"
})
,
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Body Dysmorphic Disorder\n\n* **Major Depressive Disorder:** The most common comorbid condition; onset of MDD typically follows BDD onset.\n* **Social Anxiety Disorder:** Commonly comorbid; social avoidance in BDD is often secondary to appearance-related shame.\n* **Obsessive-Compulsive Disorder (OCD):** Frequently comorbid, given the overlapping obsessive and compulsive features of both conditions.\n* **Substance-Related Disorders:** Including misuse of anabolic-androgenic steroids, particularly in the muscle dysmorphia subtype of BDD.",
    metadata={
        "disorder": "Body Dysmorphic Disorder",
        "category": "Obsessive-Compulsive and Related Disorders",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
),
Document(
page_content="CHUNK 1 — OVERVIEW\n\nHoarding Disorder\n\nCore Diagnostic Feature:\n- Hoarding Disorder is characterized by persistent difficulty discarding or parting with possessions regardless of their actual value.\n\nEssential Diagnostic Characteristics:\n1. Persistent difficulty discarding possessions.\n2. Difficulty discarding is due to a perceived need to save items.\n3. Distress occurs when attempting to discard possessions.\n4. Accumulation of possessions leads to cluttered living areas.\n5. Clutter compromises the intended use of living spaces.\n\nNature of Hoarded Items:\n- Common items saved include:\n  - Newspapers\n  - Magazines\n  - Clothing\n  - Bags\n  - Books\n  - Mail\n  - Papers\n- Any object may be hoarded, including items of real value.\n\nClutter Characteristics:\n- Objects accumulate in disorganized piles.\n- Active living spaces (kitchen, bedroom, hallway, tables, floors) become unusable.\n- Living spaces may appear uncluttered only due to intervention by third parties (e.g., family members, cleaners, authorities).\n\nSpecifier — Excessive Acquisition:\n- Approximately 80–90% of individuals with hoarding disorder also show excessive acquisition.\n- Common forms:\n  - Excessive buying\n  - Acquiring free items\n  - Collecting discarded items\n\nInsight Specifiers:\n1. With good or fair insight — Individual recognizes hoarding behaviors are problematic.\n2. With poor insight — Individual believes hoarding behaviors are not problematic despite evidence.\n3. With absent insight/delusional beliefs — Individual is completely convinced hoarding behaviors are not problematic.\n\nCourse Characteristics:\n- Symptoms typically begin during adolescence.\n- Symptoms progressively worsen across decades.\n- The course is usually chronic without treatment.",
metadata={
"disorder":"Hoarding Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"overview",
"retrieval_priority":"high"
}
),

Document(
page_content="CHUNK 2 — SYMPTOM LIST\n\nDiagnostic Threshold:\n- ALL of the following symptoms (Criteria A–C) must be present.\n\nSymptom 1:\n- Persistent difficulty discarding or parting with possessions, regardless of their actual value.\n\nSymptom 2:\n- The difficulty discarding possessions is due to:\n  1. A perceived need to save the items.\n  2. Distress associated with discarding them.\n\nSymptom 3:\n- Difficulty discarding possessions results in accumulation of possessions that:\n  1. Congest and clutter active living areas.\n  2. Substantially compromise the intended use of those areas.\n\nAdditional Diagnostic Clarification:\n- Active living areas include:\n  - Kitchens\n  - Bedrooms\n  - Living rooms\n  - Hallways\n  - Floors\n  - Tables\n\n- If living areas are uncluttered, it must be due to interventions by third parties such as:\n  - Family members\n  - Cleaners\n  - Authorities\n\nAssociated Behavioral Features:\n- Individuals intentionally save possessions.\n- Distress occurs when discarding is attempted.\n\nCommon emotional responses when discarding is attempted:\n- Anxiety\n- Frustration\n- Regret\n- Sadness\n- Guilt\n\nCommon reasons for saving items:\n- Perceived utility\n- Aesthetic value\n- Strong sentimental attachment\n- Fear of losing important information\n- Feeling responsible for the fate of possessions",
metadata={
"disorder":"Hoarding Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"symptom_list",
"retrieval_priority":"high"
}
),

Document(
page_content="CHUNK 3 — DURATION REQUIREMENTS\n\nDSM Duration Rule:\n- DSM-5 does not specify an explicit minimum duration requirement for hoarding disorder.\n\nPersistence Requirement:\n- Symptoms must be persistent and long-standing rather than temporary clutter caused by life events.\n\nCourse Characteristics:\n1. Symptoms often emerge during adolescence (approximately ages 15–19).\n2. Symptoms begin interfering with daily functioning by the mid-20s.\n3. Clinically significant impairment commonly appears by the mid-30s.\n\nCourse Pattern:\n- Severity typically increases with each decade of life.\n- Symptoms usually worsen after age 30.\n\nCourse Type:\n- The disorder is typically chronic.\n- Few individuals report a waxing and waning course.\n\nDiagnostic Clarification:\n- Temporary clutter due to life circumstances (e.g., moving, inheritance, relocation) does not meet diagnostic criteria.",
metadata={
"disorder":"Hoarding Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"duration",
"retrieval_priority":"high"
}
),

Document(
page_content="CHUNK 4 — IMPAIRMENT REQUIREMENT\n\nDiagnostic Requirement:\n- Hoarding symptoms must cause clinically significant distress OR impairment in functioning.\n\nFunctional Domains Affected:\n1. Social functioning\n2. Occupational functioning\n3. Other important areas of functioning\n4. Ability to maintain a safe environment for self and others\n\nExamples of Functional Impairment:\n- Difficulty moving through the home\n- Inability to cook due to cluttered kitchen\n- Inability to sleep in bed due to accumulated items\n- Inability to sit on furniture due to clutter\n\nAdditional Functional Consequences:\n- Impaired personal hygiene\n- Difficulty cleaning the home\n- Disruption of household utilities due to restricted access\n\nSafety Risks Associated With Severe Hoarding:\n- Fire hazards\n- Increased risk of falls\n- Unsanitary living conditions\n- Health hazards\n\nSocial and Environmental Consequences:\n- Strained family relationships\n- Conflict with neighbors\n- Legal problems with housing authorities\n- Eviction risk\n\nDiagnostic Clarification:\n- In individuals with poor insight, distress may not be self-reported but impairment is observable by others.\n- Attempts by others to remove possessions typically produce high distress.",
metadata={
"disorder":"Hoarding Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"impairment",
"retrieval_priority":"high"
}
),

Document(
page_content="CHUNK 5 — EXCLUSION CRITERIA\n\nThe diagnosis of Hoarding Disorder must NOT be made if the symptoms are better explained by the following conditions:\n\nMedical Condition Exclusion:\n- Symptoms attributable to another medical condition such as:\n  - Brain injury\n  - Cerebrovascular disease\n  - Central nervous system infection\n  - Neurogenetic disorders (e.g., Prader-Willi syndrome)\n\nNeurodevelopmental Disorder Exclusion:\n- Accumulation due to neurodevelopmental disorders such as:\n  - Autism spectrum disorder\n  - Intellectual developmental disorder\n\nPsychotic Disorder Exclusion:\n- Accumulation caused by:\n  - Delusions\n  - Negative symptoms\n  - Schizophrenia spectrum disorders\n\nMood Disorder Exclusion:\n- Accumulation resulting from:\n  - Decreased energy\n  - Psychomotor retardation\n  - Major depressive episode\n\nObsessive-Compulsive Disorder Exclusion:\n- Accumulation caused by obsessions or compulsions in OCD such as:\n  - Contamination fears\n  - Harm fears\n  - Ritual avoidance\n\nNeurocognitive Disorder Exclusion:\n- Accumulation caused by degenerative disorders such as:\n  - Alzheimer's disease\n  - Frontotemporal neurocognitive disorder\n\nDiagnostic Clarification:\n- Hoarding Disorder may be diagnosed alongside OCD if hoarding symptoms are independent from OCD symptoms.\n\nSubstance Exclusion:\n- Symptoms must not be attributable to the physiological effects of substances.",
metadata={
"disorder":"Hoarding Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"exclusion",
"retrieval_priority":"high"
}
),

Document(
page_content="CHUNK 6 — DIFFERENTIAL DIAGNOSIS\n\nNormative Collecting:\n- Organized collecting behavior.\n- Items are systematically arranged.\n- No significant clutter or impairment.\n\nObsessive-Compulsive Disorder:\n- Hoarding occurs due to specific obsessions or compulsions.\n- Items may be bizarre (e.g., trash, feces, urine, nail clippings, rotten food).\n- Excessive acquisition is uncommon.\n\nMajor Depressive Disorder:\n- Accumulation due to fatigue, low energy, or psychomotor slowing.\n\nSchizophrenia Spectrum and Other Psychotic Disorders:\n- Accumulation driven by delusions or hallucinations.\n\nNeurocognitive Disorders:\n- Accumulation occurs after onset of degenerative conditions such as:\n  - Alzheimer's disease\n  - Frontotemporal neurocognitive disorder\n\nNeurodevelopmental Disorders:\n- Accumulation due to restricted interests or cognitive deficits (e.g., autism spectrum disorder).\n\nMedical Conditions:\n- Hoarding-like behaviors resulting from neurological damage such as:\n  - Traumatic brain injury\n  - Brain tumor surgery\n  - Cerebrovascular disease\n\nDiagnostic Clarification:\n- In Hoarding Disorder, saving possessions is intentional and associated with distress when discarding is attempted.",
metadata={
"disorder":"Hoarding Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"differential",
"retrieval_priority":"medium"
}
),

Document(
page_content="CHUNK 7 — ASSOCIATED SUICIDE RISK\n\nDSM-5 Information:\n- No specific suicide risk criterion is included in the diagnostic criteria for Hoarding Disorder.\n\nAssociated Clinical Risks:\n- Hoarding Disorder is associated with multiple psychosocial stressors that may increase psychological distress.\n\nIndirect Risk Factors:\n1. Severe functional impairment.\n2. Social isolation.\n3. Family conflict.\n4. Legal or housing problems (e.g., eviction).\n\nHealth and Safety Risks Related to Severe Hoarding:\n- Fire hazards\n- Falling injuries\n- Unsanitary living conditions\n- Physical health complications\n\nClinical Note:\n- Suicide risk is not a defining diagnostic feature of Hoarding Disorder but may be evaluated in the presence of comorbid conditions such as major depressive disorder.",
metadata={
"disorder":"Hoarding Disorder",
"category":"Obsessive-Compulsive and Related Disorders",
"section":"suicide_risk",
"retrieval_priority":"low"
}
),
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Hoarding Disorder\n\n* **Comorbid Mood or Anxiety Disorder (~75%):** The majority of individuals with hoarding disorder have at least one comorbid mood or anxiety disorder.\n* **Major Depressive Disorder (30–50%):** The most common comorbid condition in hoarding disorder.\n* **Social Anxiety Disorder:** Frequently comorbid; social isolation and shame related to hoarding may contribute.\n* **Generalized Anxiety Disorder (GAD):** Commonly co-occurring, particularly around concerns about possessions and safety.\n* **Obsessive-Compulsive Disorder (~20%):** Approximately one in five individuals with hoarding disorder also meets criteria for OCD.",
    metadata={
        "disorder": "Hoarding Disorder",
        "category": "Obsessive-Compulsive and Related Disorders",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
),Document(
    page_content= "## CHUNK 1 — OVERVIEW: Excoriation (Skin-Picking) Disorder\n\nExcoriation (skin-picking) disorder is characterized by recurrent picking of one's own skin, leading to skin lesions. It is associated with repeated attempts to decrease or stop the behavior and results in clinically significant distress or impairment in social, occupational, or other important areas of functioning.\n\n**Core Features:**\n* Recurrent skin picking (usually face, arms, hands, or multiple sites).\n* Skin picking causes noticeable skin lesions.\n* Individuals spend significant time picking or thinking about picking.\n* Repeated attempts to decrease or stop skin picking.\n* Behavior persists despite distress or functional impairment.\n* Often chronic with waxing and waning course; onset typically in adolescence.", 
    metadata= { "disorder": "Excoriation (Skin-Picking) Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "overview", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 2 — SYMPTOM LIST: Excoriation (Skin-Picking) Disorder\n\n**Diagnostic Threshold:** \nAll core criteria (A–B) must be met for diagnosis.\n\n**Symptoms:**\n1. **Recurrent skin picking** resulting in skin lesions.\n2. **Repeated attempts to decrease or stop** skin picking.\n3. **Clinically significant distress or impairment** in social, occupational, or other important areas of functioning caused by skin picking.\n\n**Additional Observed Features:**\n* Skin picking may involve fingernails or instruments (tweezers, pins).\n* Picking can occur on healthy skin, minor irregularities, or scabs.\n* Skin picking may be preceded by tension or triggered by anxiety/boredom.\n* Relief, gratification, or pleasure often follows picking.\n* Behavior may be automatic (without awareness) or focused (with tension and relief).\n* Duration can be several hours per day, persisting for months or years.", 
    metadata= { "disorder": "Excoriation (Skin-Picking) Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "symptom_list", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Excoriation (Skin-Picking) Disorder\n\n* No strict minimum duration is specified in DSM; behavior is **chronic**, often **months to years**.\n* Skin picking occurs **daily or for several hours per day** in many individuals.\n* Course may wax and wane over time, with periods of remission and exacerbation.", 
    metadata= { "disorder": "Excoriation (Skin-Picking) Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "duration", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENT: Excoriation (Skin-Picking) Disorder\n\n* The skin picking must cause **clinically significant distress** (e.g., embarrassment, shame, loss of control).\n* Must cause **impairment in functioning**, including:\n  - Social avoidance\n  - Occupational interference\n  - Academic difficulties\n  - Impaired leisure activities\n* Functional consequences may include tissue damage, scarring, infection, and rarely synovitis.", 
    metadata= { "disorder": "Excoriation (Skin-Picking) Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "impairment", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 5 — EXCLUSION CRITERIA: Excoriation (Skin-Picking) Disorder\n\n* **Substance Exclusion:** Not attributable to physiological effects of substances (e.g., cocaine).\n* **Medical Condition Exclusion:** Not attributable to another medical condition (e.g., scabies, dermatological conditions without independent skin-picking behavior).\n* **Psychotic Disorder Exclusion:** Not better explained by delusions or tactile hallucinations.\n* **Other OCD-Related Disorder Exclusion:** Not better explained by body dysmorphic disorder (picking to improve perceived defect) or excessive washing compulsions in OCD.\n* **Stereotypic Movement Disorder:** Excluded if onset is in early developmental period with repetitive self-injurious behavior.\n* **Nonsuicidal Self-Injury:** Excluded if skin picking is primarily intended to harm oneself.", 
    metadata= { "disorder": "Excoriation (Skin-Picking) Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "exclusion", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Excoriation (Skin-Picking) Disorder\n\n* **Psychotic Disorders:** Picking due to delusions (e.g., parasitosis) or tactile hallucinations.\n* **Other OCD and Related Disorders:** OCD with contamination/washing compulsions; body dysmorphic disorder with skin picking.\n* **Neurodevelopmental Disorders:** Stereotypic movement disorder, Prader-Willi syndrome, or tics in Tourette's disorder.\n* **Dermatitis Artefacta / Factitious Dermatitis:** Self-induced lesions with denial of behavior.\n* **Nonsuicidal Self-Injury:** Skin picking with intention to harm oneself.\n* **Other Medical Conditions:** Dermatological conditions causing scratching (e.g., scabies, acne excoriée).\n* **Substance/Medication-Induced Disorders:** Skin picking due to substances (e.g., cocaine) or medications.", 
    metadata= { "disorder": "Excoriation (Skin-Picking) Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "differential", "retrieval_priority": "medium" }
),
Document(
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Excoriation (Skin-Picking) Disorder\n\n* **Suicidal risk is not a core feature** of excoriation disorder.\n* Comorbid mental disorders (e.g., major depressive disorder) may increase suicide risk.\n* Assessment of suicide risk should be considered if comorbid depression or other high-risk features are present.", 
    metadata= { "disorder": "Excoriation (Skin-Picking) Disorder", "category": "Obsessive-Compulsive and Related Disorders", "section": "suicide_risk", "retrieval_priority": "low" }
),
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Excoriation (Skin-Picking) Disorder\n\n* **Obsessive-Compulsive Disorder (OCD):** Frequently comorbid; skin picking shares features of repetitive, compulsive behavior with OCD.\n* **Trichotillomania (Hair-Pulling Disorder):** Frequently comorbid; both are body-focused repetitive behavior disorders.\n* **Major Depressive Disorder:** Commonly comorbid; more prevalent in women with excoriation disorder.\n* **Other Body-Focused Repetitive Behaviors:** Nail biting, lip biting, and similar behaviors may co-occur and may warrant additional diagnosis.",
    metadata={
        "disorder": "Excoriation (Skin-Picking) Disorder",
        "category": "Obsessive-Compulsive and Related Disorders",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
),
Document(
page_content="## CHUNK 1 — OVERVIEW: Trichotillomania (Hair-Pulling Disorder)\n\nDisorder Name: Trichotillomania (Hair-Pulling Disorder)\nDSM Code: F63.3\nCategory: Obsessive Compulsive Disorder\n\n**Core Diagnostic Concept:**\nTrichotillomania is characterized by recurrent pulling out of one's own hair that results in hair loss, accompanied by repeated attempts to decrease or stop the behavior and clinically significant distress or impairment.\n\n**Core Diagnostic Features:**\n- Recurrent pulling out of one's own hair.\n- Observable hair loss resulting from the pulling behavior.\n- Repeated attempts to decrease or stop hair pulling.\n- Clinically significant distress or impairment in functioning.\n\n**Typical Behavioral Characteristics:**\n- Hair may be pulled from any body region where hair grows.\n- Most common sites:\n  - Scalp\n  - Eyebrows\n  - Eyelashes\n- Less common sites:\n  - Axillary hair\n  - Facial hair\n  - Pubic hair\n  - Perirectal hair\n\n**Pattern of Hair Pulling:**\n- Hair pulling may occur:\n  - In brief episodes scattered throughout the day\n  - In sustained episodes lasting several hours\n- The behavior may persist for months or years if untreated.\n\n**Associated Behavioral Rituals:**\n- Searching for a specific hair (texture or color).\n- Pulling hair in a particular way (e.g., extracting the root intact).\n- Examining or manipulating the pulled hair.\n- Rolling hair between fingers.\n- Pulling hair between teeth.\n- Biting hair into pieces.\n- Swallowing hair (trichophagia).\n\n**Emotional and Sensory Features:**\n- Hair pulling may be triggered by:\n  - Anxiety\n  - Boredom\n- Individuals may experience:\n  - Increasing tension before pulling\n  - Gratification, pleasure, or relief after pulling\n- Some individuals report an itch-like or tingling sensation relieved by pulling.\n\n**Behavioral Awareness:**\n- Hair pulling may occur:\n  - With focused awareness\n  - Automatically without full awareness\n  - With a combination of both behavioral styles.\n\n**Common Additional Body-Focused Repetitive Behaviors:**\n- Skin picking\n- Nail biting\n- Lip chewing",
metadata={
"disorder": "Trichotillomania (Hair-Pulling Disorder)",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "overview",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Trichotillomania (Hair-Pulling Disorder)\n\n**Diagnostic Threshold Rule:**\nAll of the following criteria (A–E) must be present for diagnosis.\n\n**Mandatory Core Symptoms:**\n1. **Recurrent pulling out of one's own hair**, resulting in **hair loss**.\n\n2. **Repeated attempts to decrease or stop hair pulling.**\n\n3. Hair pulling behavior results in **clinically significant distress** or **impairment in functioning**.\n\n4. The hair pulling or hair loss **is not attributable to another medical condition** (e.g., dermatological conditions).\n\n5. The hair pulling behavior **is not better explained by another mental disorder**.\n\n**Examples of Behavioral Contexts (Not Required Diagnostic Symptoms):**\n- Pulling single hairs from multiple locations across the scalp.\n- Pulling from eyebrows or eyelashes resulting in partial or complete absence of hair.\n- Pulling in both focused and automatic behavioral patterns.",
metadata={
"disorder": "Trichotillomania (Hair-Pulling Disorder)",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "symptom_list",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Trichotillomania (Hair-Pulling Disorder)\n\n**DSM Duration Specification:**\n- The DSM-5 diagnostic criteria for Trichotillomania **do not specify a fixed minimum duration** (e.g., no requirement such as ≥2 weeks or ≥6 months).\n\n**Clinical Course Characteristics:**\n- Hair pulling episodes may occur:\n  - Briefly throughout the day, or\n  - During prolonged episodes lasting hours.\n\n- The behavior may **persist for months or years**.\n\n- The disorder typically follows a **chronic course** with possible waxing and waning when untreated.\n\n- Onset most commonly occurs **around puberty**.\n\n**Diagnostic Duration Rule:**\n- Diagnosis requires **recurrent hair pulling behavior** with repeated attempts to stop and associated impairment, regardless of a fixed minimum duration.",
metadata={
"disorder": "Trichotillomania (Hair-Pulling Disorder)",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "duration",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Trichotillomania (Hair-Pulling Disorder)\n\n**Required Diagnostic Condition:**\n- The hair pulling behavior must cause **clinically significant distress OR impairment**.\n\n**Forms of Distress:**\n- Feelings of loss of control over the behavior\n- Embarrassment\n- Shame\n\n**Functional Impairment Domains:**\n- Social functioning\n- Occupational functioning\n- Academic functioning\n- Leisure or recreational functioning\n\n**Examples of Functional Impact:**\n- Avoidance of work or school.\n- Avoidance of social or public situations.\n- Attempts to conceal hair loss using makeup, wigs, scarves, or clothing.\n\n**Additional Functional Consequences:**\n- Possible irreversible hair growth damage.\n- Medical complications in rare cases:\n  - Carpal tunnel syndrome\n  - Back, neck, or shoulder pain\n  - Dental damage from hair biting\n  - Trichobezoars resulting from hair swallowing.",
metadata={
"disorder": "Trichotillomania (Hair-Pulling Disorder)",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "impairment",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Trichotillomania (Hair-Pulling Disorder)\n\nDiagnosis must **NOT** be made if hair pulling is better explained by any of the following:\n\n**Medical Condition Exclusion:**\n- Hair pulling or hair loss attributable to a medical condition, including:\n  - Dermatological conditions\n  - Alopecia areata\n  - Androgenic alopecia\n  - Telogen effluvium\n  - Chronic discoid lupus erythematosus\n  - Lichen planopilaris\n  - Central centrifugal cicatricial alopecia\n  - Pseudopelade\n  - Folliculitis decalvans\n  - Dissecting folliculitis\n  - Acne keloidalis nuchae\n\n**Mental Disorder Exclusion:**\n- Hair pulling better explained by another mental disorder, including:\n  - Body dysmorphic disorder (removal of perceived ugly or abnormal hair)\n  - Obsessive-compulsive disorder symmetry rituals\n\n**Psychotic Disorder Exclusion:**\n- Hair removal performed in response to **delusions or hallucinations** in psychotic disorders.\n\n**Substance-Related Exclusion:**\n- Hair-pulling behavior attributable to the **physiological effects of substances**, including stimulant-induced symptoms.\n\n**Diagnostic Rule:**\n- Trichotillomania is diagnosed **only when hair pulling is not attributable to another medical condition, substance effect, or mental disorder.**",
metadata={
"disorder": "Trichotillomania (Hair-Pulling Disorder)",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "exclusion",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Trichotillomania (Hair-Pulling Disorder)\n\n**1. Normative Hair Removal or Manipulation**\n- Hair removal performed solely for cosmetic purposes.\n- Twisting or playing with hair without resulting hair loss.\n- Hair biting without hair pulling.\n\n**2. Obsessive-Compulsive Disorder (OCD)**\n- Hair pulling performed as part of symmetry rituals.\n- Behavior driven by intrusive obsessions requiring ritualized responses.\n\n**3. Body Dysmorphic Disorder (BDD)**\n- Hair removal performed to correct a perceived defect or abnormality in appearance.\n\n**4. Stereotypic Movement Disorder**\n- Repetitive behaviors such as:\n  - Head banging\n  - Hand biting\n  - Arm biting\n  - Hair pulling\n- Occurs particularly in individuals with:\n  - Intellectual developmental disorder\n  - Autism spectrum disorder\n\n**5. Psychotic Disorders**\n- Hair removal due to hallucinations or delusional beliefs.\n\n**6. Dermatological and Medical Causes of Alopecia**\n- Non-scarring alopecia conditions:\n  - Alopecia areata\n  - Androgenic alopecia\n  - Telogen effluvium\n\n- Scarring alopecia conditions:\n  - Chronic discoid lupus erythematosus\n  - Lichen planopilaris\n  - Central centrifugal cicatricial alopecia\n\n**7. Substance-Related Disorders**\n- Hair-pulling behavior exacerbated or induced by substances such as stimulants.",
metadata={
"disorder": "Trichotillomania (Hair-Pulling Disorder)",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "differential",
"retrieval_priority": "medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Trichotillomania (Hair-Pulling Disorder)\n\n**DSM Suicide Risk Information:**\n- The DSM diagnostic criteria for Trichotillomania **do not include suicidal ideation or suicide attempts as core diagnostic symptoms.**\n\n**Associated Psychological Distress:**\n- Individuals may experience:\n  - Embarrassment\n  - Shame\n  - Feelings of loss of control\n\n**Comorbid Mental Disorders (Which May Increase Suicide Risk):**\n- Major Depressive Disorder\n- Excoriation (Skin-Picking) Disorder\n- Other body-focused repetitive behavior disorders\n\n**Clinical Consideration:**\n- Suicide risk may be indirectly elevated due to comorbid depressive disorders rather than the hair-pulling behavior itself.",
metadata={
"disorder": "Trichotillomania (Hair-Pulling Disorder)",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "suicide_risk",
"retrieval_priority": "low"
}
)
,
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Trichotillomania (Hair-Pulling Disorder)\n\n* **Major Depressive Disorder:** The most common comorbid condition in trichotillomania; elevated rates of depressive symptoms are consistently reported.\n* **Excoriation (Skin-Picking) Disorder:** Frequently comorbid; both are body-focused repetitive behavior disorders sharing similar behavioral patterns.\n* **Other Body-Focused Repetitive Behaviors:** Nail biting, lip biting, and cheek chewing may co-occur and may warrant an additional diagnosis.",
    metadata={
        "disorder": "Trichotillomania (Hair-Pulling Disorder)",
        "category": "Obsessive-Compulsive and Related Disorders",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
)
,
Document(
page_content="## CHUNK 1 — OVERVIEW: Substance/Medication-Induced Obsessive-Compulsive and Related Disorder\n\nSubstance/Medication-Induced Obsessive-Compulsive and Related Disorder is characterized by prominent symptoms of obsessive-compulsive and related disorders that are directly attributable to the physiological effects of a substance (e.g., drug of abuse, toxin, or medication).\n\n**Core Clinical Concept:**\n* The obsessive-compulsive and related symptoms are caused by **substance intoxication, withdrawal, medication exposure, or toxin exposure**.\n\n**Core Features:**\n* Prominent obsessive-compulsive and related symptoms dominate the clinical presentation.\n* Symptoms develop **during or soon after substance intoxication, withdrawal, or medication exposure**.\n* The involved substance or medication must be **capable of producing obsessive-compulsive and related symptoms**.\n* The disturbance is severe enough to warrant **independent clinical attention**.\n\n**Common Symptom Types Observed:**\n* Obsessions\n* Compulsions\n* Skin picking (excoriation)\n* Hair pulling (trichotillomania)\n* Other body-focused repetitive behaviors\n\n**Common Substances Associated With Symptoms:**\n* Stimulants (e.g., cocaine, amphetamine-type substances)\n* Other or unknown substances\n* Certain medications\n* Heavy metals or toxins\n\n**Clinical Course:**\n* Symptoms typically **improve or remit within days to weeks** after cessation of the substance or medication.\n* Persistence of symptoms **beyond about 1 month after intoxication or withdrawal** suggests a primary obsessive-compulsive disorder rather than a substance-induced condition.",
metadata={
"disorder": "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "overview",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 2 — SYMPTOM LIST: Substance/Medication-Induced Obsessive-Compulsive and Related Disorder\n\n**Diagnostic Threshold Rule:**\nAll diagnostic criteria **A–E must be met** for diagnosis.\n\n**Mandatory Core Symptoms:**\n1. **Obsessions** predominate in the clinical picture.\n2. **Compulsions** predominate in the clinical picture.\n3. **Skin picking (excoriation)** behaviors occur and predominate.\n4. **Hair pulling (trichotillomania)** behaviors occur and predominate.\n5. **Other body-focused repetitive behaviors** characteristic of obsessive-compulsive and related disorders.\n6. **Other symptoms characteristic of obsessive-compulsive and related disorders** predominate.\n\n**Substance Relationship Requirements:**\n7. Symptoms **develop during or soon after substance intoxication or withdrawal**, or after exposure to or withdrawal from a medication.\n8. The **substance or medication is capable of producing obsessive-compulsive and related symptoms**.\n\n**Diagnostic Rule:**\n* The obsessive-compulsive and related symptoms must **predominate in the clinical picture** and be sufficiently severe to warrant clinical attention.",
metadata={
"disorder": "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "symptom_list",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 3 — DURATION REQUIREMENTS: Substance/Medication-Induced Obsessive-Compulsive and Related Disorder\n\n**Onset Timing Requirement:**\n* Symptoms must develop **during or soon after** one of the following:\n  - Substance intoxication\n  - Substance withdrawal\n  - Exposure to a medication\n  - Withdrawal from a medication\n\n**Expected Course:**\n* Symptoms typically **improve or remit within days to weeks** after cessation of the substance or medication.\n\n**Persistence Rule (Diagnostic Clarification):**\n* If symptoms **persist for a substantial period of time (approximately ≥1 month)** after the cessation of acute withdrawal or severe intoxication, an **independent obsessive-compulsive and related disorder should be considered**.\n\n**Clinical Timing Principle:**\n* The disorder must show a **temporal relationship between substance exposure and symptom onset**.",
metadata={
"disorder": "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "duration",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Substance/Medication-Induced Obsessive-Compulsive and Related Disorder\n\n**Required Diagnostic Condition:**\n* The disturbance must cause **clinically significant distress OR impairment**.\n\n**Domains of Functional Impairment:**\n* Social functioning\n* Occupational functioning\n* Academic functioning\n* Other important areas of functioning\n\n**Clinical Attention Rule:**\n* The obsessive-compulsive and related symptoms must **predominate in the clinical picture** and be **sufficiently severe to warrant independent clinical attention** beyond typical substance intoxication or withdrawal effects.",
metadata={
"disorder": "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "impairment",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 5 — EXCLUSION CRITERIA: Substance/Medication-Induced Obsessive-Compulsive and Related Disorder\n\nDiagnosis must **NOT** be made if any of the following conditions apply:\n\n**Primary Obsessive-Compulsive Disorder Exclusion:**\n* Symptoms **precede the onset of substance or medication use**.\n* Symptoms **persist for ≥1 month after cessation of severe intoxication or withdrawal**.\n* There is a **history of recurrent non–substance-related obsessive-compulsive episodes**.\n\n**Delirium Exclusion:**\n* The disturbance **does not occur exclusively during the course of delirium**.\n\n**Substance Intoxication/Withdrawal Exclusion:**\n* If symptoms are **typical of normal intoxication or withdrawal**, a diagnosis of intoxication or withdrawal alone may suffice.\n\n**Medical Condition Exclusion:**\n* If symptoms are **better explained by another medical condition**, diagnose **obsessive-compulsive and related disorder due to another medical condition** instead.\n\n**Diagnostic Attribution Rule:**\n* Diagnosis requires evidence from **history, physical examination, or laboratory findings** that the substance or medication is responsible for the symptoms.",
metadata={
"disorder": "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "exclusion",
"retrieval_priority": "high"
}
),

Document(
page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Substance/Medication-Induced Obsessive-Compulsive and Related Disorder\n\n**1. Substance Intoxication or Substance Withdrawal**\n* Obsessive-compulsive symptoms may occur during intoxication or withdrawal.\n* Diagnosis of intoxication or withdrawal alone may suffice if symptoms are **typical and not excessively severe**.\n* Diagnose substance/medication-induced obsessive-compulsive disorder only if symptoms are **in excess of those usually associated with intoxication or withdrawal** and require independent clinical attention.\n\n**2. Primary Obsessive-Compulsive and Related Disorder**\n* Symptoms may resemble primary OCD or related disorders.\n* Primary disorder is suggested when:\n  - Symptoms **precede substance use**.\n  - Symptoms **persist ≥1 month after cessation of intoxication or withdrawal**.\n  - Individual has **history of non-substance-related episodes**.\n\n**3. Obsessive-Compulsive and Related Disorder Due to Another Medical Condition**\n* Symptoms may be caused by **another medical condition rather than substance effects**.\n* Clinical history and medical evaluation help determine causation.\n\n**4. Delirium**\n* If obsessive-compulsive symptoms occur **exclusively during delirium**, they are considered part of delirium and **not diagnosed separately**.",
metadata={
"disorder": "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "differential",
"retrieval_priority": "medium"
}
),

Document(
page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Substance/Medication-Induced Obsessive-Compulsive and Related Disorder\n\n**DSM Diagnostic Criteria:**\n* Suicidal ideation or suicide attempts **are not included as core diagnostic criteria** for this disorder.\n\n**Potential Indirect Risk Factors:**\n* Psychological distress related to severe obsessive-compulsive symptoms.\n* Co-occurring substance use disorders.\n* Comorbid psychiatric disorders.\n\n**Clinical Consideration:**\n* Suicide risk assessment may be necessary when severe distress, substance misuse, or comorbid mental disorders are present.",
metadata={
"disorder": "Substance/Medication-Induced Obsessive-Compulsive and Related Disorder",
"category": "Obsessive-Compulsive and Related Disorders",
"section": "suicide_risk",
"retrieval_priority": "low"
}
),
Document(
    page_content= "## CHUNK 1 — OVERVIEW: Bipolar I Disorder\n\nBipolar I disorder is characterized by a clinical course of recurring mood episodes, including manic, depressive, and hypomanic episodes. Diagnosis requires at least one lifetime manic episode.\n\n**Core Features:**\n* Presence of at least **one manic episode** (Criteria A–D under Manic Episode).\n* Manic episodes may be preceded or followed by hypomanic or major depressive episodes, but these are not required for diagnosis.\n* Manic episode is a distinct period of abnormally and persistently elevated, expansive, or irritable mood and increased energy or activity.\n* Represents a noticeable change from the individual's usual behavior.",
    metadata= { "disorder": "Bipolar I Disorder", "category": "Bipolar and related Disorder", "section": "overview", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 2 — SYMPTOM LIST: Bipolar I Disorder\n\n### Manic Episode Diagnostic Threshold:\n* Mood: Abnormally elevated, expansive, or irritable + increased activity or energy.\n* Duration: At least **1 week**, most of the day, nearly every day (any duration if hospitalization is necessary).\n* **Symptoms Required:** ≥3 symptoms if mood is elevated/expansive; ≥4 if mood is only irritable.\n\n**Symptoms:**\n1. Inflated self-esteem or grandiosity.\n2. Decreased need for sleep (e.g., rested after 3 hours).\n3. More talkative than usual or pressure to keep talking.\n4. Flight of ideas or racing thoughts.\n5. Distractibility (attention drawn to irrelevant stimuli).\n6. Increase in goal-directed activity (social, work/school, sexual) or psychomotor agitation.\n7. Excessive involvement in activities with high potential for painful consequences (e.g., unrestrained buying sprees, sexual indiscretions, foolish business investments).",
    metadata= { "disorder": "Bipolar I Disorder", "category": "Bipolar and related Disorder", "section": "symptom_list", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Bipolar I Disorder\n\n* **Manic Episode:** Symptoms must persist for at least **1 week**, most of the day, nearly every day; any duration if hospitalization is necessary.\n* **Hypomanic Episode:** Symptoms must persist for at least **4 consecutive days**, most of the day, nearly every day.\n* **Major Depressive Episode:** Symptoms must persist for at least **2 weeks**, most of the day, nearly every day.",
    metadata= { "disorder": "Bipolar I Disorder", "category": "Bipolar and related Disorder", "section": "duration", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENT: Bipolar I Disorder\n\n* Manic episode must cause **marked impairment** in social or occupational functioning, **necessitate hospitalization** to prevent harm to self or others, or involve **psychotic features**.\n* Hypomanic episodes do **not** cause marked impairment or require hospitalization.\n* Major depressive episodes must cause **clinically significant distress or impairment** in social, occupational, or other important areas of functioning.",
    metadata= { "disorder": "Bipolar I Disorder", "category": "Bipolar and related Disorder", "section": "impairment", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 5 — EXCLUSION CRITERIA: Bipolar I Disorder\n\n* **Substance Exclusion:** Not attributable to physiological effects of substances (drugs of abuse, medications, or treatments).\n* **Medical Condition Exclusion:** Not attributable to another medical condition.\n* **Bipolar/Psychotic Exclusion:**\n  - Not better explained by schizoaffective disorder.\n  - Not superimposed on schizophrenia, schizophreniform disorder, delusional disorder, or other specified/unspecified schizophrenia spectrum disorders.\n* **Treatment-Induced Episodes:** Manic or hypomanic episodes emerging during antidepressant treatment that persist beyond physiological effect count as valid episodes.",
    metadata= { "disorder": "Bipolar I Disorder", "category": "Bipolar and related Disorder", "section": "exclusion", "retrieval_priority": "high" }
),
Document(
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Bipolar I Disorder\n\n* **Major Depressive Disorder:** Distinguish by assessing history of mania/hypomania, early onset, family history, numerous episodes, psychotic features, or antidepressant-induced mania.\n* **Other Bipolar Disorders:** Bipolar II, cyclothymic disorder, or other specified bipolar disorders differ by absence of any manic episodes.\n* **Anxiety Disorder:** GAD, panic disorder, PTSD; differentiate based on episodic nature and symptom triggers.\n* **Bipolar Due to Another Medical Condition:** Manic episodes must be judged to result from medical conditions (e.g., Cushing’s disease, multiple sclerosis).\n* **Substance/Medication-Induced Bipolar Disorder:** Mania attributable to substances or medications (e.g., stimulants, steroids) rather than primary bipolar I disorder.\n* **Schizoaffective Disorder:** Psychotic symptoms occurring outside mood episodes suggest schizoaffective disorder.\n* **ADHD:** Persistent inattention, hyperactivity, impulsivity with onset by age 12; contrast with episodic mania.\n* **Disruptive Mood Dysregulation Disorder (DMDD):** Persistent severe irritability, not discrete manic/hypomanic episodes.\n* **Personality Disorders:** Mood lability and impulsivity must represent distinct episodes beyond baseline.",
    metadata= { "disorder": "Bipolar I Disorder", "category": "Bipolar and related Disorder", "section": "differential", "retrieval_priority": "medium" }
),
Document(
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Bipolar I Disorder\n\n* **Lifetime Risk:** Estimated 20–30 times greater than the general population.\n* **Mortality:** Approximately 5%–6% die by suicide.\n* **Gender Differences:** Suicide attempts higher in women; completed suicide more common in men.\n* **Risk Factors:** Past suicide attempts, high percent days spent depressed, comorbid alcohol use disorder increase risk.\n* **Clinical Implication:** Suicide risk assessment is essential during all mood episodes, especially with mixed or depressive features.",
    metadata= { "disorder": "Bipolar I Disorder", "category": "Bipolar and related Disorder", "section": "suicide_risk", "retrieval_priority": "low" }
),
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Bipolar I Disorder\n\n* **Comorbidity Pattern:** Co-occurring mental disorders are the norm in bipolar I disorder; the majority of individuals have a history of three or more comorbid disorders.\n* **Anxiety Disorders:** The most frequently comorbid condition group in bipolar I disorder.\n* **Alcohol Use Disorder:** One of the most common individual comorbidities.\n* **Other Substance Use Disorders:** Substantially elevated rates compared to the general population; sociocultural factors (e.g., cultural prohibitions) influence prevalence.\n* **ADHD:** Frequently comorbid with bipolar I disorder.\n* **Personality Disorders:** Borderline, schizotypal, and antisocial personality disorders are frequently comorbid. The relationship with borderline personality disorder may reflect shared phenomenology, vulnerability factors, and early childhood adversity.\n* **Medical Conditions:** High rates of serious comorbid medical conditions contributing to shortened life expectancy, including cardiovascular disease, autoimmune diseases, obstructive sleep apnea, metabolic syndrome, and migraine.\n* **Overweight/Obesity:** A particular concern; associated with poor treatment outcomes.",
    metadata={
        "disorder": "Bipolar I Disorder",
        "category": "Bipolar and related Disorder",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
),
Document(
    page_content= "## CHUNK 1 — OVERVIEW: Bipolar II Disorder\n\nBipolar II disorder is characterized by a recurrent course of mood episodes, including at least one hypomanic episode and at least one major depressive episode. Hypomanic episodes involve distinct periods of abnormally elevated, expansive, or irritable mood with increased activity or energy, lasting at least 4 consecutive days. Major depressive episodes involve depressed mood or markedly diminished interest/pleasure lasting at least 2 weeks.\n\n**Core Features:**\n* Recurrent mood episodes: hypomanic and major depressive.\n* Hypomanic episodes do not cause marked impairment or psychosis.\n* Major depressive episodes cause significant distress or impairment.\n* No history of full manic episodes.\n* Mood episodes not attributable to substances, medications, or medical conditions.\n* Episodes not better explained by schizoaffective disorder or other psychotic disorders.", 
    metadata= {
        "disorder": "Bipolar II Disorder",
        "category": "Bipolar and related Disorder",
        "section": "overview",
        "retrieval_priority": "high"
    }
),

Document(
    page_content= "## CHUNK 2 — SYMPTOM LIST: Bipolar II Disorder\n\n### Hypomanic Episode\n**Diagnostic Threshold:**\n* At least **3 symptoms** must be present (or **4 if mood is only irritable**).\n* Symptoms must represent a noticeable change from usual behavior and be present to a significant degree.\n\n**Symptoms:**\n1. Inflated self-esteem or grandiosity.\n2. Decreased need for sleep (e.g., feels rested after only 3 hours).\n3. More talkative than usual or pressure to keep talking.\n4. Flight of ideas or subjective experience of racing thoughts.\n5. Distractibility (attention easily drawn to irrelevant stimuli).\n6. Increase in goal-directed activity (socially, at work/school, or sexually) or psychomotor agitation.\n7. Excessive involvement in activities with high potential for painful consequences (e.g., unrestrained spending, sexual indiscretions, foolish business investments).\n\n### Major Depressive Episode\n**Diagnostic Threshold:**\n* At least **5 symptoms** must be present during the same 2-week period.\n* At least one symptom must be **depressed mood** or **loss of interest/pleasure**.\n\n**Symptoms:**\n1. Depressed mood most of the day, nearly every day.\n2. Markedly diminished interest or pleasure in all or almost all activities.\n3. Significant weight loss/gain or decrease/increase in appetite.\n4. Insomnia or hypersomnia nearly every day.\n5. Psychomotor agitation or retardation nearly every day.\n6. Fatigue or loss of energy nearly every day.\n7. Feelings of worthlessness or excessive/inappropriate guilt nearly every day.\n8. Diminished ability to think, concentrate, or indecisiveness nearly every day.\n9. Recurrent thoughts of death, suicidal ideation, or suicide attempts.", 
    metadata= {
        "disorder": "Bipolar II Disorder",
        "category": "Bipolar and related Disorder",
        "section": "symptom_list",
        "retrieval_priority": "high"
    }
),

Document(
    page_content= "## CHUNK 3 — DURATION REQUIREMENTS: Bipolar II Disorder\n\n* **Hypomanic Episode:** Symptoms must persist for at least **4 consecutive days**, present most of the day, nearly every day.\n* **Major Depressive Episode:** Symptoms must persist for at least **2 weeks**, present most of the day, nearly every day.\n* Recurrence: Bipolar II disorder is highly recurrent; over 50% experience a new episode within a year of the first episode.", 
    metadata= {
        "disorder": "Bipolar II Disorder",
        "category": "Bipolar and related Disorder",
        "section": "duration",
        "retrieval_priority": "high"
    }
),

Document(
    page_content= "## CHUNK 4 — IMPAIRMENT REQUIREMENT: Bipolar II Disorder\n\n* Major depressive episodes or frequent mood alternations cause **clinically significant distress** or **impairment** in social, occupational, or other important areas.\n* Hypomanic episodes do **not** cause marked impairment or require hospitalization.\n* Functional impairment may be cumulative due to recurrent depressive episodes and unpredictable mood changes.", 
    metadata= {
        "disorder": "Bipolar II Disorder",
        "category": "Bipolar and related Disorder",
        "section": "impairment",
        "retrieval_priority": "high"
    }
),

Document(
    page_content= "## CHUNK 5 — EXCLUSION CRITERIA: Bipolar II Disorder\n\n* **Manic Episode Exclusion:** There has never been a manic episode.\n* **Substance/Medication Exclusion:** Mood episodes are not attributable to drugs, medications, or other treatments.\n* **Medical Condition Exclusion:** Mood episodes are not due to another medical condition.\n* **Psychotic/Bipolar Disorder Exclusion:** Symptoms not better explained by schizoaffective disorder, schizophrenia, schizophreniform disorder, delusional disorder, or other specified/unspecified psychotic disorders.", 
    metadata= {
        "disorder": "Bipolar II Disorder",
        "category": "Bipolar and related Disorder",
        "section": "exclusion",
        "retrieval_priority": "high"
    }
),

Document(
    page_content= "## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Bipolar II Disorder\n\n* **Major Depressive Disorder:** Distinguished by absence of hypomanic episodes.\n* **Cyclothymic Disorder:** Does not meet full criteria for hypomanic or major depressive episodes.\n* **Schizophrenia:** Diagnosed if major depressive episodes are only minority of time during active psychotic symptoms.\n* **Schizoaffective Disorder:** Psychotic symptoms occur ≥2 weeks without major depressive episode.\n* **Bipolar/Related Disorder due to Medical Condition:** Direct physiological consequence of a medical condition.\n* **Substance/Medication-Induced Bipolar Disorder:** Mood episodes caused by substance/medication.\n* **ADHD:** Overlapping symptoms, distinguish by episodic change over baseline.\n* **Personality Disorders:** Distinguish by presence of distinct episodes and deviation from baseline.\n* **Other Bipolar Disorders:** Differentiate from bipolar I and other specified/unspecified bipolar disorders.", 
    metadata= {
        "disorder": "Bipolar II Disorder",
        "category": "Bipolar and related Disorder",
        "section": "differential",
        "retrieval_priority": "medium"
    }
),

Document(
    page_content= "## CHUNK 7 — ASSOCIATED SUICIDE RISK: Bipolar II Disorder\n\n* **Lifetime Risk:** Approximately one-third of individuals report a lifetime suicide attempt.\n* **Risk Comparison:** Similar rates of attempts and suicide deaths compared to bipolar I disorder; higher than general population.\n* **Clinical Concern:** Time spent in depressive episodes increases risk; lethality of attempts may be higher than bipolar I disorder.\n* **Genetic Risk:** First-degree relatives of bipolar II probands have a 6.5-fold higher risk of suicide compared to first-degree relatives of bipolar I probands.", 
    metadata= {
        "disorder": "Bipolar II Disorder",
        "category": "Bipolar and related Disorder",
        "section": "suicide_risk",
        "retrieval_priority": "low"
    }
),
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Bipolar II Disorder\n\n* **Comorbidity Pattern:** Bipolar II disorder is more often than not associated with one or more co-occurring mental disorders.\n* **Anxiety Disorders (75%):** The most common comorbidity; social anxiety disorder (38%), specific phobia (36%), and generalized anxiety disorder (30%) are most prevalent. Lifetime anxiety comorbidity rates are similar between bipolar I and II but worsen illness course. Anxiety disorders tend to associate most with depressive symptoms.\n* **Multiple Comorbidities (~60%):** Approximately 60% of individuals with bipolar II have three or more co-occurring mental disorders.\n* **Substance Use Disorders:** Elevated above general population rates; closely linked to mood states rather than following an independent course. Most common: alcohol use disorder (42%) and cannabis use disorder (20%). Substance use disorders associate moderately with hypomanic symptoms.\n* **PTSD:** Lower rates of comorbid PTSD compared to individuals with bipolar I disorder.\n* **Eating Disorders (~14%):** At least one lifetime eating disorder; binge-eating disorder is more common than bulimia nervosa or anorexia nervosa.\n* **Premenstrual Syndrome / PMDD:** Common in women with bipolar II disorder; associated with more severe bipolar mood symptoms and lability.\n* **Medical Conditions:** Comorbid cardiovascular disease, migraine, and autoimmune disorders complicate course and prognosis.",
    metadata={
        "disorder": "Bipolar II Disorder",
        "category": "Bipolar and related Disorder",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
),

# ── Cyclothymic Disorder ─────────────────────────────────────────────────────
Document(
    page_content="## CHUNK 1 — OVERVIEW: Cyclothymic Disorder\n\nCyclothymic Disorder is characterized by a chronic, fluctuating mood disturbance involving numerous periods of hypomanic symptoms and periods of depressive symptoms that **do not meet full criteria** for a hypomanic episode or a major depressive episode.\n\n**Core Diagnostic Features:**\n* Chronic alternating periods of hypomanic and depressive symptoms.\n* Symptoms are subthreshold — insufficient in number, severity, pervasiveness, or duration to qualify as a full hypomanic or major depressive episode.\n* Symptoms must be present for at least half the time over the 2-year period (1 year in children/adolescents), with no symptom-free interval lasting more than 2 months.\n* Criteria for a major depressive, manic, or hypomanic episode have **never** been met.\n\n**Course Characteristics:**\n* Usual onset: adolescence or early adult life.\n* Course is typically insidious and persistent.\n* 15%–50% risk of subsequently developing Bipolar I or Bipolar II disorder.\n* Vast majority of youth with cyclothymic disorder experience mood symptom onset before age 10.\n\n**Specifier:**\n* With anxious distress.",
    metadata={
        "disorder": "Cyclothymic Disorder",
        "category": "Bipolar and related Disorder",
        "section": "overview",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 2 — SYMPTOM LIST: Cyclothymic Disorder\n\n**Diagnostic Criteria (All of A–F must be met):**\n\n**Criterion A — Core Symptom Pattern:**\n* Numerous periods with **hypomanic symptoms** (insufficient to meet full hypomanic episode criteria).\n* Numerous periods with **depressive symptoms** (insufficient to meet full major depressive episode criteria).\n* Required duration: at least 2 years (adults); at least 1 year (children and adolescents).\n\n**Criterion B — Persistence Rule:**\n* During the 2-year period (1 year for children/adolescents), Criterion A symptoms must be present for **at least half the time**.\n* The individual must not have been symptom-free for more than **2 months at a time**.\n\n**Criterion C — Episode Exclusion:**\n* Criteria for a **major depressive episode, manic episode, or hypomanic episode** have **never** been met.\n* If such an episode occurs after the initial 2-year period, the diagnosis changes to MDD, Bipolar I, or other specified bipolar and related disorder.\n\n**Hypomanic Symptom Domains (subthreshold):**\n* Elevated, expansive, or irritable mood.\n* Increased energy or activity.\n* Decreased need for sleep, talkativeness, distractibility, increased goal-directed activity, impulsivity.\n\n**Depressive Symptom Domains (subthreshold):**\n* Depressed or sad mood.\n* Reduced energy, interest, concentration.\n* Sleep or appetite changes.",
    metadata={
        "disorder": "Cyclothymic Disorder",
        "category": "Bipolar and related Disorder",
        "section": "symptom_list",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 3 — DURATION REQUIREMENTS: Cyclothymic Disorder\n\n* **Minimum Duration:**\n  - Adults: at least **2 years** of fluctuating hypomanic and depressive symptoms.\n  - Children and adolescents: at least **1 year**.\n\n* **Persistence Requirement:**\n  - Symptoms must be present for **at least half** the required time period.\n  - The individual must not have been completely symptom-free for more than **2 consecutive months** during this period.\n\n* **Course:**\n  - Onset is usually insidious; course is typically persistent and chronic.\n  - Conversion risk to Bipolar I or II is 15–50%; diagnostic conversion rates are higher in youth than adults.\n  - Late-onset persistent hypomanic/depressive symptoms must be differentiated from bipolar/depressive disorder due to another medical condition before assigning the cyclothymic disorder diagnosis.",
    metadata={
        "disorder": "Cyclothymic Disorder",
        "category": "Bipolar and related Disorder",
        "section": "duration",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Cyclothymic Disorder\n\n* Symptoms must cause **clinically significant distress** or **impairment** in social, occupational, or other important areas of functioning (Criterion F).\n\n**Sources of Impairment:**\n* The symptoms themselves (subthreshold hypomanic and depressive mood changes).\n* The **pattern of unpredictability and inconsistency** in mood, which negatively affects interpersonal functioning and role performance.\n\n**Common Functional Consequences:**\n* Impaired family and occupational role functioning.\n* Disrupted relationships due to mood inconsistency.\n* Academic or work performance difficulties.\n\n**Note:** Some individuals may function particularly well during periods of hypomania; however, over the prolonged course of the disorder, clinically significant distress or impairment must be demonstrated overall.",
    metadata={
        "disorder": "Cyclothymic Disorder",
        "category": "Bipolar and related Disorder",
        "section": "impairment",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 5 — EXCLUSION CRITERIA: Cyclothymic Disorder\n\n**Criterion C — Full Episode Exclusion:**\n* Criteria for a **major depressive episode, manic episode, or hypomanic episode** must never have been met. If such an episode subsequently develops, the cyclothymic disorder diagnosis is dropped and replaced by the appropriate diagnosis.\n\n**Criterion D — Psychotic Disorder Exclusion:**\n* The mood pattern must not be better explained by:\n  - Schizoaffective disorder\n  - Schizophrenia\n  - Schizophreniform disorder\n  - Delusional disorder\n  - Other specified or unspecified schizophrenia spectrum and psychotic disorders\n* In those contexts, mood symptoms are considered associated features of the psychotic disorder.\n\n**Criterion E — Substance/Medical Exclusion:**\n* Symptoms must not be attributable to the **physiological effects of a substance** (e.g., drug of abuse, medication) or **another medical condition** (e.g., hyperthyroidism).",
    metadata={
        "disorder": "Cyclothymic Disorder",
        "category": "Bipolar and related Disorder",
        "section": "exclusion",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Cyclothymic Disorder\n\n* **Bipolar and Related Disorder Due to Another Medical Condition:** Diagnosed when a specific, usually chronic medical condition (e.g., hyperthyroidism) directly causes the mood disturbance. If the symptoms are the psychological rather than physiological consequence, both cyclothymic disorder and the medical condition are coded separately.\n\n* **Substance/Medication-Induced Bipolar Disorder:** Stimulants and other substances may produce frequent mood swings resembling cyclothymia. These typically resolve following cessation of the substance.\n\n* **Bipolar I or II Disorder with Rapid Cycling:** Both involve frequent marked mood shifts. The key distinction is that cyclothymic disorder requires that criteria for a full major depressive, manic, or hypomanic episode have **never** been met, whereas rapid cycling requires full mood episodes.\n\n* **Borderline Personality Disorder:** Associated with recurrent, brief shifts in mood that may resemble cyclothymia. Key distinction: mood instability in BPD occurs in domains of anxiety, irritability, and sadness; elation, euphoria, and increased energy are not characteristic of BPD. Both diagnoses can be made if criteria for both are met.",
    metadata={
        "disorder": "Cyclothymic Disorder",
        "category": "Bipolar and related Disorder",
        "section": "differential",
        "retrieval_priority": "medium"
    }
),
Document(
    page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Cyclothymic Disorder\n\n* Suicide risk is **not explicitly specified** as a core diagnostic feature of cyclothymic disorder in DSM criteria.\n* The chronic, fluctuating mood course and associated distress may contribute to psychological vulnerability.\n* **Elevated Conversion Risk:** 15–50% of individuals with cyclothymic disorder subsequently develop bipolar I or bipolar II disorder, both of which carry substantially elevated suicide risk.\n* **Comorbid Conditions:** Co-occurring substance use disorders (which are common in cyclothymic disorder) independently increase suicide risk.\n* **Clinical Implication:** Suicide risk should be evaluated in all individuals with cyclothymic disorder, particularly during depressive symptom periods and if conversion to a full mood episode occurs.",
    metadata={
        "disorder": "Cyclothymic Disorder",
        "category": "Bipolar and related Disorder",
        "section": "suicide_risk",
        "retrieval_priority": "low"
    }
),
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Cyclothymic Disorder\n\n* **Substance-Related Disorders:** May be present in individuals with cyclothymic disorder; there may also be an increased familial risk of substance-related disorders.\n* **Sleep Disorders:** Difficulties in initiating and maintaining sleep are commonly co-occurring.\n* **High Overall Comorbidity in Youth:** Rates of comorbid psychiatric disorders in children with cyclothymic disorder treated in outpatient psychiatric settings are greater than those in children with disruptive behavior/ADHD disorders and similar to those in children with bipolar I or II disorder.\n* **Family History:** Major depressive disorder, bipolar I disorder, and bipolar II disorder are more common among first-degree biological relatives of individuals with cyclothymic disorder than in the general population.",
    metadata={
        "disorder": "Cyclothymic Disorder",
        "category": "Bipolar and related Disorder",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
),

# ── Substance/Medication-Induced Bipolar and Related Disorder ────────────────
Document(
    page_content="## CHUNK 1 — OVERVIEW: Substance/Medication-Induced Bipolar and Related Disorder\n\nSubstance/Medication-Induced Bipolar and Related Disorder is characterized by a prominent and persistent disturbance in mood — specifically **abnormally elevated, expansive, or irritable mood with abnormally increased activity or energy** — that is directly attributable to the physiological effects of a substance, medication, or toxin.\n\n**Core Diagnostic Concept:**\n* Manic or hypomanic-like symptoms are causally linked to substance intoxication, substance withdrawal, medication initiation, medication change, or medication withdrawal.\n* The disturbance must be sufficiently severe and clinically prominent to warrant independent clinical attention.\n\n**Substances and Medications Commonly Implicated:**\n* Stimulants (cocaine, amphetamine-type substances) — most common\n* Phencyclidine (PCP)\n* Corticosteroids and immunosuppressant medications\n* Other or unknown substances\n\n**Key Diagnostic Clarifications:**\n* Hypomania or mania persisting **beyond the physiological effects** of antidepressant treatment indicates true bipolar disorder, NOT substance/medication-induced disorder.\n* Simple agitation or irritability from antidepressants, without a full manic/hypomanic syndrome, does not meet diagnostic criteria.\n* This diagnosis is made **instead of** substance intoxication or withdrawal when manic/hypomanic symptoms predominate and warrant clinical attention.",
    metadata={
        "disorder": "Substance/Medication-Induced Bipolar and Related Disorder",
        "category": "Bipolar and related Disorder",
        "section": "overview",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 2 — SYMPTOM LIST: Substance/Medication-Induced Bipolar and Related Disorder\n\n**Diagnostic Criteria (All of A–E must be met):**\n\n**Criterion A — Core Symptom:**\n* Prominent and persistent disturbance in mood characterized by **abnormally elevated, expansive, or irritable mood** AND **abnormally increased activity or energy**.\n\n**Criterion B — Evidence of Substance/Medication Causation (both B1 and B2):**\n1. Symptoms developed **during or soon after** substance intoxication or withdrawal, or after exposure to or withdrawal from a medication.\n2. The involved substance/medication is **capable of producing** the symptoms in Criterion A.\n\n**Criterion C — Not Better Explained by Independent Bipolar Disorder:**\n* Evidence of an independent bipolar disorder includes:\n  - Symptoms **preceded** the onset of substance/medication use.\n  - Symptoms **persist for a substantial period** (approximately ≥1 month) after cessation of acute withdrawal or severe intoxication.\n  - History of recurrent non-substance-related manic episodes.\n\n**Criterion D — Delirium Exclusion:**\n* The disturbance does **not** occur exclusively during a delirium.\n\n**Criterion E — Functional Impairment:**\n* Symptoms cause clinically significant distress or impairment in social, occupational, or other important areas of functioning.",
    metadata={
        "disorder": "Substance/Medication-Induced Bipolar and Related Disorder",
        "category": "Bipolar and related Disorder",
        "section": "symptom_list",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 3 — DURATION REQUIREMENTS: Substance/Medication-Induced Bipolar and Related Disorder\n\n* No fixed minimum duration is specified in DSM criteria for this disorder.\n\n**Onset Timing by Substance:**\n* **Phencyclidine:** Initial delirium with affective features developing within hours to a few days after ingestion/inhalation; transitions to atypically appearing manic or mixed state.\n* **Stimulants (cocaine, amphetamines):** Response occurs within minutes to 1 hour; episode is very brief, typically resolving over 1–2 days.\n* **Corticosteroids / Immunosuppressants:** Mania (or mixed/depressed state) usually follows several days of ingestion; higher doses carry greater risk.\n\n**Resolution Expectation:**\n* Symptoms typically resolve shortly after cessation of the substance or medication.\n* If symptoms persist for a **substantial period (approximately ≥1 month)** beyond acute withdrawal or severe intoxication, an **independent bipolar and related disorder** should be considered instead.",
    metadata={
        "disorder": "Substance/Medication-Induced Bipolar and Related Disorder",
        "category": "Bipolar and related Disorder",
        "section": "duration",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 4 — IMPAIRMENT REQUIREMENT: Substance/Medication-Induced Bipolar and Related Disorder\n\n* Criterion E requires that the mood disturbance causes **clinically significant distress or impairment** in social, occupational, or other important areas of functioning.\n\n* This diagnosis is made **instead of** substance intoxication or substance withdrawal only when the manic/hypomanic symptoms:\n  - **Predominate** in the clinical picture, AND\n  - Are **sufficiently severe** to warrant independent clinical attention.\n\n**Functional Consequences (Context-Dependent):**\n* Impaired judgment and behavior during induced manic state.\n* Risk of self-harm or harm to others during elevated/irritable mood episodes.\n* Occupational, social, or legal consequences related to the episode.\n* Complications related to the underlying substance use disorder.",
    metadata={
        "disorder": "Substance/Medication-Induced Bipolar and Related Disorder",
        "category": "Bipolar and related Disorder",
        "section": "impairment",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 5 — EXCLUSION CRITERIA: Substance/Medication-Induced Bipolar and Related Disorder\n\n**Criterion C — Independent Bipolar Disorder Exclusion:**\n* The disturbance must not be better explained by a bipolar or related disorder that is **not** substance/medication-induced. Evidence for an independent disorder includes:\n  - Symptoms preceded the substance/medication use.\n  - Symptoms persist ≥1 month after cessation of acute withdrawal/severe intoxication.\n  - History of recurrent non-substance-related manic episodes.\n\n**Criterion D — Delirium Exclusion:**\n* The disturbance must not occur **exclusively** during the course of a delirium.\n\n**Antidepressant / ECT Exception:**\n* A full manic episode emerging during antidepressant treatment or ECT and persisting beyond the physiological effect of the treatment is evidence for **Bipolar I disorder**, not substance-induced disorder.\n* A full hypomanic episode under the same conditions, preceded by a major depressive episode, is evidence for **Bipolar II disorder**.\n\n**Symptom Specificity Rule:**\n* Simple irritability, edginess, or agitation during antidepressant treatment — without a full manic or hypomanic syndrome — is **not** sufficient for this diagnosis.",
    metadata={
        "disorder": "Substance/Medication-Induced Bipolar and Related Disorder",
        "category": "Bipolar and related Disorder",
        "section": "exclusion",
        "retrieval_priority": "high"
    }
),
Document(
    page_content="## CHUNK 6 — DIFFERENTIAL DIAGNOSIS: Substance/Medication-Induced Bipolar and Related Disorder\n\n* **Bipolar I Disorder:** A full manic episode that emerges during antidepressant treatment and persists beyond the physiological effect of that treatment is sufficient evidence for Bipolar I, not substance-induced disorder.\n\n* **Bipolar II Disorder:** A full hypomanic episode emerging during antidepressant treatment, persisting beyond physiological effects, and preceded by a major depressive episode is sufficient evidence for Bipolar II.\n\n* **Substance Intoxication:** Euphoria, irritability, and increased energy occurring during intoxication (e.g., stimulants) are covered by the substance-specific intoxication diagnosis unless the manic/hypomanic symptoms predominate and are severe enough to warrant independent attention.\n\n* **Substance Withdrawal:** Mood changes during withdrawal (e.g., cannabis withdrawal) are covered by the withdrawal diagnosis unless manic/hypomanic symptoms predominate.\n\n* **Delirium:** When manic symptoms occur exclusively during a delirium (e.g., phencyclidine ingestion), the delirium diagnosis takes precedence; the substance-induced bipolar diagnosis is not made.\n\n* **Medication Side Effects:** Edginess or agitation from antidepressants or psychotropic drugs is distinct from a full bipolar syndrome and is insufficient for this diagnosis.",
    metadata={
        "disorder": "Substance/Medication-Induced Bipolar and Related Disorder",
        "category": "Bipolar and related Disorder",
        "section": "differential",
        "retrieval_priority": "medium"
    }
),
Document(
    page_content="## CHUNK 7 — ASSOCIATED SUICIDE RISK: Substance/Medication-Induced Bipolar and Related Disorder\n\n* Suicide risk is **not specified as a core diagnostic criterion** of substance/medication-induced bipolar and related disorder.\n* Risk is primarily driven by the **underlying substance use disorder** and the **induced manic/mixed state**, both of which independently increase impulsivity and risk-taking.\n* **Phencyclidine-induced states** may carry particularly elevated risk due to associated delirium, aggression, and extreme behavioral disorganization.\n* **Clinical Implication:** Suicide risk and safety should be assessed during any induced manic or mixed episode, particularly when judgment and impulse control are severely impaired.",
    metadata={
        "disorder": "Substance/Medication-Induced Bipolar and Related Disorder",
        "category": "Bipolar and related Disorder",
        "section": "suicide_risk",
        "retrieval_priority": "low"
    }
),
Document(
    page_content="## CHUNK 8 — COMORBIDITY: Substance/Medication-Induced Bipolar and Related Disorder\n\n* **Substance Use Disorders:** The primary comorbidity; the disorder is by definition caused by a substance or medication. Comorbidities reflect the underlying substance use context.\n  - Illicit stimulant use (cocaine, amphetamines): associated substance use disorder comorbidities.\n  - Phencyclidine use: associated substance use disorder comorbidities.\n  - Prescribed stimulants (diversion): comorbidities related to ADHD or other indications.\n* **Medical Conditions (Steroid/Immunosuppressant Context):** Comorbidities are those medical indications requiring steroid or immunosuppressant treatment (e.g., autoimmune diseases, organ transplantation, inflammatory conditions).\n* **Delirium:** Can occur before or alongside manic symptoms in individuals using phencyclidine or those prescribed steroid/immunosuppressant medications.\n* **Underlying Bipolar Diathesis:** Substance/medication-induced manic symptoms may unmask an earlier, undiagnosed predisposition to bipolar disorder in some individuals.",
    metadata={
        "disorder": "Substance/Medication-Induced Bipolar and Related Disorder",
        "category": "Bipolar and related Disorder",
        "section": "comorbidity",
        "chunk_type": "comorbidity",
        "retrieval_priority": "high"
    }
),

Document(
    page_content= "## Depressive Disorders Overview and Classification\n\nDepressive disorders include disruptive mood dysregulation disorder, major depressive disorder (including major depressive episode), persistent depressive disorder, premenstrual dysphoric disorder, substance/medication-induced depressive disorder, depressive disorder due to another medical condition, other specified depressive disorder, and unspecified depressive disorder. The common feature across these disorders is the presence of sad, empty, or irritable mood, accompanied by somatic and cognitive changes that significantly affect the individual’s capacity to function. Differences among them involve duration, timing, or presumed etiology.\n\n**Disruptive Mood Dysregulation Disorder:** Observed in children up to 12 years with persistent irritability and frequent episodes of extreme behavioral dyscontrol. Typically predicts later unipolar depressive or anxiety disorders rather than bipolar disorders.\n\n**Major Depressive Disorder (MDD):** Characterized by discrete episodes lasting at least 2 weeks (often longer) with changes in affect, cognition, and neurovegetative functions. Recurrent in most cases, but a single episode can be diagnosed. Differentiation from normal grief is essential; bereavement-related depressive episodes are more severe if combined with MDD.\n\n**Persistent Depressive Disorder:** Chronic mood disturbance lasting at least 2 years in adults or 1 year in children, encompassing chronic major depression and dysthymia.\n\n**Premenstrual Dysphoric Disorder:** Mood symptoms begin after ovulation and remit within days of menses, significantly impacting functioning.\n\n**Substance/Medication-Induced and Medical Condition-Related Depressive Disorders:** Depression-like symptoms caused by substances of abuse, prescribed medications, or medical conditions, recognized under these specific diagnoses.\n\n**Key Classification Criteria:**\n* Core symptom: persistent sad, empty, or irritable mood.\n* Functional impact: significant impairment in social, occupational, or other areas.\n* Differentiation based on duration, timing, etiology, and context (e.g., bereavement, medical conditions, substance exposure).\n* Age-specific considerations for disruptive mood dysregulation disorder.\n* Consideration of recurrent vs. single episodes in MDD, chronicity in persistent depressive disorder, and cyclicity in premenstrual dysphoric disorder.", 
    metadata={ "category": "Depressive Disorders", "section": "overview_summary_criteria", "retrieval_priority": "high" }
)
,Document(
    page_content= "## Anxiety Disorder Overview and Classification\n\nAnxiety disorders are characterized by excessive fear and anxiety with related behavioral disturbances. Fear is the emotional response to real or perceived imminent threat, often with autonomic arousal and escape behaviors, whereas anxiety is anticipation of future threat, often with muscle tension, vigilance, and avoidant behaviors. Panic attacks can occur across anxiety and other mental disorders and indicate severity and prognosis.\n\n**Core Features:** Persistent, excessive fear or anxiety; behavioral avoidance; significant impairment in social, occupational, or other important areas; differentiation from developmentally normative fear or stress-induced anxiety. Duration typically 6 months or more, though shorter in children (e.g., separation anxiety disorder, selective mutism). Symptoms must not be attributable to substances, medications, medical conditions, or better explained by another mental disorder.\n\n**Disorders:**\n- **Separation Anxiety Disorder:** Excessive fear or anxiety about separation from attachment figures, worry about harm to attachment figures, nightmares, and physical distress, onset in childhood but may persist into adulthood.\n- **Selective Mutism:** Consistent failure to speak in social situations despite speaking in other contexts, impairing academic/occupational achievement or social communication.\n- **Specific Phobia:** Fear or avoidance of circumscribed objects or situations; immediate and disproportionate response; subtypes include animal, natural environment, blood-injection-injury, situational, and other.\n- **Social Anxiety Disorder:** Fear of social interactions or performance situations due to concern about negative evaluation, embarrassment, humiliation, or rejection.\n- **Panic Disorder:** Recurrent unexpected panic attacks, with persistent concern about future attacks or maladaptive behavior changes; panic attacks may be expected or unexpected.\n- **Agoraphobia:** Fear and avoidance of multiple situations (e.g., public transportation, open spaces, enclosed places, crowds, being outside alone) due to concern about escape difficulty or lack of help.\n- **Generalized Anxiety Disorder:** Persistent, excessive anxiety and worry about multiple domains, difficult to control; accompanied by restlessness, fatigue, concentration problems, irritability, muscle tension, and sleep disturbance.\n- **Substance/Medication-Induced Anxiety Disorder:** Anxiety due to intoxication, withdrawal, or medication effects.\n- **Anxiety Disorder Due to Another Medical Condition:** Anxiety as a physiological consequence of a medical condition.\n\n**Clinical Notes:** Anxiety disorders differ in feared objects, situations, and associated cognition; highly comorbid with each other; may develop in childhood and persist if untreated; more frequent in females (~2:1 ratio). Disorder-specific scales are available for severity assessment. Certain anxiety disorders (panic disorder, generalized anxiety disorder, specific phobia) are associated with increased suicide risk.",
    metadata={ "category": "Anxiety Disorder", "section": "overview_summary_criteria", "retrieval_priority": "high" }
),Document(
    page_content= "## Obsessive-Compulsive and Related Disorders Overview and Classification\n\nObsessive-compulsive and related disorders (OCRDs) include OCD, body dysmorphic disorder, hoarding disorder, trichotillomania (hair-pulling disorder), excoriation (skin-picking) disorder, substance/medication-induced OCRD, OCRD due to another medical condition, other specified OCRD (e.g., nail biting, lip biting, cheek chewing, obsessional jealousy, olfactory reference disorder), and unspecified OCRD. These disorders are characterized by obsessions, preoccupations, or recurrent body-focused repetitive behaviors, accompanied by attempts to resist or control the behaviors, and cause clinically significant distress or impairment.\n\n**Core Features:** Excessive, persistent, or developmentally inappropriate obsessions, preoccupations, or body-focused repetitive behaviors; repeated attempts to decrease or stop behaviors; distress or impairment in social, occupational, or other important areas; differentiation from normative developmental rituals or subclinical symptoms.\n\n**Disorders:**\n- **Obsessive-Compulsive Disorder (OCD):** Presence of obsessions (intrusive, unwanted thoughts, urges, or images) and/or compulsions (repetitive behaviors or mental acts performed to reduce distress or follow rigid rules). Symptom dimensions include cleaning (contamination), symmetry (ordering/repeating/counting), forbidden/taboo thoughts (aggressive, sexual, religious), and harm (checking). Tic-related specifier applies when a history of tics is present.\n- **Body Dysmorphic Disorder (BDD):** Preoccupation with perceived defects in appearance; repetitive behaviors (mirror checking, grooming, skin picking, reassurance seeking) or mental acts (appearance comparison). Muscle dysmorphia is a subtype involving belief of insufficient muscularity. Excludes concerns about body fat or weight in eating disorders.\n- **Hoarding Disorder:** Persistent difficulty discarding possessions due to perceived need to save items and distress, leading to clutter that impairs living space. Excessive acquisition may be present.\n- **Trichotillomania:** Recurrent hair pulling resulting in hair loss, with repeated attempts to reduce or stop behavior.\n- **Excoriation (Skin-Picking) Disorder:** Recurrent skin picking causing lesions, with repeated attempts to reduce or stop behavior. May be preceded by tension and relieved by picking; varying awareness (focused vs. automatic).\n- **Substance/Medication-Induced OCRD:** Symptoms characteristic of OCRDs occurring during substance intoxication, withdrawal, or medication exposure.\n- **OCRD Due to Another Medical Condition:** Symptoms caused directly by another medical condition.\n- **Other Specified and Unspecified OCRDs:** Clinically significant distress or impairment from symptoms not meeting criteria for a specific OCRD, including atypical presentations or insufficient information.\n\n**Clinical Notes:** OCRDs share relatedness across disorders, warranting screening for comorbid conditions. Cognitive-component OCRDs (OCD, BDD, hoarding) include insight specifiers: good/fair, poor, or absent/delusional. Insight determines severity and treatment considerations; absent insight does not warrant a separate psychotic disorder unless delusional content extends beyond OCRD.",
    metadata={ "category": "Obsessive-Compulsive and Related Disorders", "section": "overview_summary_criteria", "retrieval_priority": "high" }
),Document(
    page_content= "## Obsessive-Compulsive and Related Disorders Overview and Classification\n\nObsessive-compulsive and related disorders (OCRDs) include OCD, body dysmorphic disorder, hoarding disorder, trichotillomania (hair-pulling disorder), excoriation (skin-picking) disorder, substance/medication-induced OCRD, OCRD due to another medical condition, other specified OCRD (e.g., nail biting, lip biting, cheek chewing, obsessional jealousy, olfactory reference disorder), and unspecified OCRD. These disorders are characterized by obsessions, preoccupations, or recurrent body-focused repetitive behaviors, accompanied by attempts to resist or control the behaviors, and cause clinically significant distress or impairment.\n\n**Core Features:** Excessive, persistent, or developmentally inappropriate obsessions, preoccupations, or body-focused repetitive behaviors; repeated attempts to decrease or stop behaviors; distress or impairment in social, occupational, or other important areas; differentiation from normative developmental rituals or subclinical symptoms.\n\n**Disorders:**\n- **Obsessive-Compulsive Disorder (OCD):** Presence of obsessions (intrusive, unwanted thoughts, urges, or images) and/or compulsions (repetitive behaviors or mental acts performed to reduce distress or follow rigid rules). Symptom dimensions include cleaning (contamination), symmetry (ordering/repeating/counting), forbidden/taboo thoughts (aggressive, sexual, religious), and harm (checking). Tic-related specifier applies when a history of tics is present.\n- **Body Dysmorphic Disorder (BDD):** Preoccupation with perceived defects in appearance; repetitive behaviors (mirror checking, grooming, skin picking, reassurance seeking) or mental acts (appearance comparison). Muscle dysmorphia is a subtype involving belief of insufficient muscularity. Excludes concerns about body fat or weight in eating disorders.\n- **Hoarding Disorder:** Persistent difficulty discarding possessions due to perceived need to save items and distress, leading to clutter that impairs living space. Excessive acquisition may be present.\n- **Trichotillomania:** Recurrent hair pulling resulting in hair loss, with repeated attempts to reduce or stop behavior.\n- **Excoriation (Skin-Picking) Disorder:** Recurrent skin picking causing lesions, with repeated attempts to reduce or stop behavior. May be preceded by tension and relieved by picking; varying awareness (focused vs. automatic).\n- **Substance/Medication-Induced OCRD:** Symptoms characteristic of OCRDs occurring during substance intoxication, withdrawal, or medication exposure.\n- **OCRD Due to Another Medical Condition:** Symptoms caused directly by another medical condition.\n- **Other Specified and Unspecified OCRDs:** Clinically significant distress or impairment from symptoms not meeting criteria for a specific OCRD, including atypical presentations or insufficient information.\n\n**Clinical Notes:** OCRDs share relatedness across disorders, warranting screening for comorbid conditions. Cognitive-component OCRDs (OCD, BDD, hoarding) include insight specifiers: good/fair, poor, or absent/delusional. Insight determines severity and treatment considerations; absent insight does not warrant a separate psychotic disorder unless delusional content extends beyond OCRD.",
    metadata={ "category": "Obsessive-Compulsive and Related Disorders", "section": "overview_summary_criteria", "retrieval_priority": "high" }
),Document(
    page_content= "## Trauma- and Stressor-Related Disorders Overview and Classification\n\nTrauma- and stressor-related disorders include reactive attachment disorder, disinhibited social engagement disorder, posttraumatic stress disorder (PTSD), acute stress disorder, prolonged grief disorder, adjustment disorders, other specified trauma- and stressor-related disorder, and unspecified trauma- and stressor-related disorder. These disorders require exposure to a traumatic or stressful event as part of the diagnostic criteria.\n\n**Core Features:** Exposure to a traumatic or stressful event is listed explicitly as a diagnostic criterion. Psychological distress following exposure varies in presentation: anxiety- or fear-based symptoms, anhedonic/dysphoric symptoms, externalizing anger/aggression, dissociative symptoms, or combinations thereof.\n\n**Disorders:**\n- **Reactive Attachment Disorder:** Inhibited, emotionally withdrawn behavior toward adult caregivers; rarely seeks or responds to comfort; results from extremes of insufficient care.\n- **Disinhibited Social Engagement Disorder:** Pattern of culturally inappropriate, overly familiar behavior with unfamiliar adults; reduced reticence; results from extremes of insufficient care.\n- **Posttraumatic Stress Disorder (PTSD):** Develops after exposure to actual/threatened death, serious injury, or sexual violence. Involves intrusion symptoms, avoidance, negative cognition/mood changes, and arousal/reactivity alterations. Duration >1 month.\n- **Acute Stress Disorder:** Similar to PTSD but occurs 3 days to 1 month after trauma exposure. Requires ≥9 symptoms from intrusion, negative mood, dissociation, avoidance, and arousal domains.\n- **Prolonged Grief Disorder:** Persistent grief response after death of someone close, with intense yearning and preoccupation exceeding culturally expected mourning.\n- **Adjustment Disorders:** Emotional or behavioral symptoms in response to identifiable stressor(s), occurring within 3 months of stressor onset.\n\n**Clinical Notes:** Trauma exposure is an explicit criterion. Presentations vary widely and may include fear-based, anhedonic, externalizing, or dissociative symptom profiles. Age-specific criteria exist for PTSD in children ≤6 years. High comorbidity with depressive, anxiety, and substance use disorders.",
    metadata={ "category": "Trauma- and Stressor-Related Disorders", "section": "overview_summary_criteria", "retrieval_priority": "high" }
),Document(
    page_content= "## Bipolar and Related Disorders Overview and Classification\n\nBipolar and related disorders include bipolar I disorder, bipolar II disorder, cyclothymic disorder, substance/medication-induced bipolar and related disorder, bipolar and related disorder due to another medical condition, other specified bipolar and related disorder, and unspecified bipolar and related disorder. These disorders are characterized by episodes of mania, hypomania, and/or major depression.\n\n**Core Features:** Distinct episodes of abnormally elevated, expansive, or irritable mood with increased energy or activity (mania/hypomania), alternating with episodes of depressed mood or loss of interest/pleasure (major depression). Bipolar I requires at least one manic episode; bipolar II requires at least one hypomanic episode and one major depressive episode.\n\n**Disorders:**\n- **Bipolar I Disorder:** At least one lifetime manic episode (≥7 days or any duration if hospitalization required). Manic episodes may be preceded or followed by hypomanic or major depressive episodes. Elevated, expansive, or irritable mood with ≥3 symptoms (≥4 if only irritable): inflated self-esteem, decreased sleep, pressured speech, racing thoughts, distractibility, increased goal-directed activity, excessive risky behavior.\n- **Bipolar II Disorder:** At least one hypomanic episode (≥4 consecutive days) and at least one major depressive episode. No history of manic episodes. Hypomania involves elevated/expansive/irritable mood with increased energy but not severe enough for hospitalization or psychosis.\n- **Cyclothymic Disorder:** Chronic (≥2 years; 1 year in children) fluctuating mood with numerous periods of hypomanic and depressive symptoms not meeting full episode criteria.\n- **Substance/Medication-Induced Bipolar:** Manic/hypomanic symptoms occurring during substance intoxication, withdrawal, or medication exposure.\n- **Bipolar Due to Another Medical Condition:** Manic/hypomanic features as a physiological consequence of a medical condition.\n\n**Clinical Notes:** Bipolar disorders bridge schizophrenia spectrum disorders and depressive disorders. High suicide risk, especially in bipolar II. Careful differentiation from unipolar depression, ADHD, and personality disorders needed. Mixed features, anxious distress, rapid cycling, and psychotic features specifiers are clinically important.",
    metadata={ "category": "Bipolar and related Disorder", "section": "overview_summary_criteria", "retrieval_priority": "high" }
)


]

# You can also create a function to get chunks
def get_all_chunks():
    return chunks

def get_chunks_by_section(section):
    return [chunk for chunk in chunks if chunk.metadata.get("section") == section]

def get_chunks_by_category(category):
    return [chunk for chunk in chunks if chunk.metadata.get("category") == category]


# Just the sections list
categorys = [c.metadata['category'] for c in chunks if 'category' in c.metadata]

# categorys with indices
categorys_with_idx = [(i, c.metadata['category']) for i, c in enumerate(chunks) if 'category' in c.metadata]

# Unique categorys and their counts
from collections import Counter
print(Counter([c.metadata['category'] for c in chunks if 'category' in c.metadata]))