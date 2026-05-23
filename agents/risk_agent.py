"""
Risk Assessment Agent v3.2 — Manchester Triage System

FIXES v3.2 (over v3.1):
✅ FIX 1: "sangre en el papel/papel higiénico" → LOW (hemorrhoids context)
✅ FIX 2: 37.x°C (subfebril) → LOW, not HIGH — only 38.5+ triggers HIGH for child
✅ FIX 3: "un poco triste hoy" → LOW + safety_sensitive=True (mild, not MEDIUM)
✅ FIX 4: blood_symptoms_non_emergency now excludes toilet paper context
✅ FIX 5: fever_high_child now requires 38.5+ not just 38
"""

import re
import time
from enum import Enum
from typing import Dict, Any, List, Tuple
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_core.prompts import PromptTemplate
from logger import HealthChatLogger


class RiskLevel(Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class TriageRules:
    """
    Evidence-based triage rules based on Manchester Triage System.
    v3.2: Fixed over-triaging of common benign presentations.
    """

    # ── HIGH RISK red flags ───────────────────────────────────────────────────
    HIGH_RISK_RULES: List[Tuple[str, List[str]]] = [

        ("fever_high_child", [
            
            r"(niño|bebé|hijo|kid|child|baby|enfant).{0,40}(fiebre|fever|fièvre).{0,20}(38\.5|39|40|alta|high)",
            r"(fiebre|fever|fièvre).{0,40}(niño|bebé|hijo|kid|child|baby|enfant).{0,20}(38\.5|39|40)",
            r"(hijo|niño|bebé).{0,20}(38\.5|39|40|41).{0,10}(grados|degrees|°)",
        ]),

        ("fever_very_high_adult", [
            r"(fiebre|fever|fièvre|temperatura|temperature).{0,20}(40|41|42|104|105)",
            r"(fiebre|fever|fièvre).{0,30}(no\s+baja|won'?t\s+(go\s+down|break)|ne\s+baisse\s+pas).{0,20}(día|day|jour)",
            r"(fiebre|fever|fièvre).{0,30}(\d+\s*días|\d+\s*days|\d+\s*jours).{0,20}(alta|high|élevée)",
        ]),

        ("severe_abdominal_pain", [
            r"(dolor|pain|douleur).{0,20}(abdominal|abdomen|barriga|estómago|stomach|ventre).{0,20}(severo|severe|intense|intenso|fort)",
            r"(dolor|pain).{0,10}(costado|lado\s+derecho|right\s+side|côté\s+droit).{0,20}(fuerte|intense|severe)",
        ]),

        ("neurological_symptoms", [
            r"(visión|vision|vue).{0,20}(doble|borrosa|double|blurred|floue).{0,20}(súbito|sudden|soudain|hoy|today)",
            r"(entumecimiento|numbness|engourdissement).{0,20}(un\s+lado|one\s+side|un\s+côté)",
            r"(debilidad|weakness|faiblesse).{0,20}(súbita|sudden|soudaine|repentina)",
            r"(confusión|confusion).{0,20}(repentina|sudden|soudaine)",
        ]),

        ("severe_allergic", [
            r"(alergia|allergy|allergie).{0,30}(severa|grave|severe|serious|sévère)",
            r"(urticaria|hives|urticaire).{0,20}(todo\s+el\s+cuerpo|whole\s+body|corps\s+entier)",
            r"(hinchazón|swelling|gonflement).{0,20}(cara|face|visage|labios|lips|lèvres)",
        ]),

        ("high_fever_prolonged", [
            r"(fiebre|fever|fièvre).{0,20}(39|40).{0,30}(2\s*días|3\s*días|dos\s*días|tres\s*días|2\s*days|3\s*days)",
            r"(39\.5|39\.7|39\.8|39\.9).{0,30}(días|days|jours)",
        ]),

        ("infection_signs", [
            r"(herida|wound|plaie|corte|cut).{0,30}(infect|pus|supura|roja\s+y\s+caliente|red\s+and\s+hot)",
            r"(líneas\s+rojas|red\s+lines|lignes\s+rouges).{0,20}(herida|wound|plaie)",
        ]),

        ("difficulty_swallowing_severe", [
            r"(dificultad|difficulty|difficulté).{0,20}(tragar|swallow|avaler).{0,20}(\d+\s*días|\d+\s*days)",
            r"(no\s+puedo|can'?t|ne\s+peut\s+pas).{0,20}(tragar|swallow|avaler)",
        ]),

        ("blood_symptoms_non_emergency", [
           
            r"(sangre|blood|sang)\s+en\s+(orina|urine|urina)",
            r"(heces|stool|selles).{0,20}(oscuras|negras|dark|black|noires)",
            r"(tosiendo|coughing|toussant).{0,20}(sangre|blood|sang)",
        ]),

        ("mental_health_crisis", [
            r"(sin\s+dormir|haven'?t\s+slept|n'?ai\s+pas\s+dormi).{0,20}(\d+\s*días|\d+\s*days)",
            r"(ataque|attack|crise).{0,20}(pánico|panic|panique).{0,20}(frecuente|frequent|souvent)",
            r"(pensamientos|thoughts|pensées).{0,20}(suicid|muerte|death|mort)",
        ]),

        ("unintentional_weight_loss", [
            r"(perdí|lost|perdu).{0,20}\d+\s*(kilos?|kg|pounds?|lbs?).{0,30}(sin|without|sans).{0,20}(dieta|diet|régime)",
            r"(pérdida|loss|perte).{0,20}(peso|weight|poids).{0,20}(involuntaria|unintentional|involontaire)",
        ]),

        ("medication_dosage_request", [
            r"(puedo|can\s+i|puis-je).{0,20}(tomar|take|prendre).{0,20}\d+",
            r"\d+\s*(ibuprofeno?|paracetamol|aspirin\w*|pastillas?|pills?)",
            r"(sobredosis|overdose|surdosage)",
            r"(mezclar|mix|mélanger).{0,20}(medicamentos|medications|médicaments)",
        ]),
    ]

    # ── MEDIUM RISK ───────────────────────────────────────────────────────────
    MEDIUM_RISK_RULES: List[Tuple[str, List[str]]] = [

        ("persistent_fever_mild", [
            r"(fiebre|fever|fièvre).{0,20}(38|38\.5).{0,30}(\d+\s*días|\d+\s*days)",
            r"(fiebre|fever|fièvre).{0,20}(días|days|jours).{0,10}(no\s+baja|persiste)",
        ]),

        ("sore_throat_days", [
            r"(dolor\s+de\s+garganta|sore\s+throat|mal\s+de\s+gorge).{0,30}(\d+\s*días|\d+\s*days|\d+\s*jours)",
            r"(garganta|throat|gorge).{0,20}(duele|hurts|fait\s+mal).{0,20}(días|days|jours)",
        ]),

        ("persistent_cough", [
            r"(tos|cough|toux).{0,20}(persistente|persistent|seca|dry|sèche).{0,30}(\d+\s*semanas?|\d+\s*weeks?)",
            r"(llevo|been|depuis).{0,20}(semana|week|semaine).{0,20}(tos|cough|toux)",
        ]),

        ("urinary_symptoms", [
            r"(ardor|burning|brûlure).{0,20}(orinar|urinate|uriner)",
            r"(infección\s+de\s+orina|urinary\s+infection|infection\s+urinaire|itu\b|uti\b)",
            r"(frecuencia|frequency|fréquence).{0,20}(orinar|urinate|uriner)",
        ]),

        ("skin_rash_days", [
            r"(sarpullido|rash|éruption).{0,20}(días|days|jours)",
            r"(manchas|spots|taches).{0,20}(piel|skin|peau).{0,20}(días|days|jours)",
            r"(picazón|itching|démangeaison).{0,20}(todo|everywhere|partout)",
        ]),

        ("headache_frequent", [
            r"(dolor\s+de\s+cabeza|headache|mal\s+de\s+tête).{0,30}(frecuente|frequent|souvent|veces\s+por\s+semana|times\s+a\s+week)",
            r"(migraña|migraine).{0,20}(frecuente|frequent|repetida|recurring)",
        ]),

        ("digestive_persistent", [
            r"(náuseas|nausea|nausées).{0,20}(días|days|jours)",
            r"(diarrea|diarrhea|diarrhée).{0,20}(\d+\s*días|\d+\s*days)",
            r"(reflujo|reflux).{0,20}(frecuente|frequent|souvent)",
        ]),

        ("chronic_condition_question", [
            r"\b(diabetes|diabète|سكري)\b",
            r"\b(hipertensión|hypertension|presión\s+alta|high\s+blood\s+pressure)\b",
            r"\b(asma|asthma|asthme)\b",
            r"\b(tiroides|thyroid|thyroïde)\b",
        ]),

        ("anxiety_sleep", [
            r"(ansiedad|anxiety|anxiété).{0,20}(no\s+puedo\s+dormir|can'?t\s+sleep|ne\s+dors\s+pas)",
            r"(insomnio|insomnia|insomnie).{0,20}(semanas?|weeks?|semaines?)",
            r"(sin\s+dormir|haven'?t\s+slept).{0,20}(días|days|jours)",
        ]),

        ("minor_injury", [
            r"(corte|cut|coupure).{0,20}(profundo|deep|profonde|puntos?|stitches?)",
            r"(torcedura|sprain|entorse).{0,20}(hinchado|swollen|gonflé)",
            r"(golpe|bump|choc).{0,20}(hinchazón|swelling|gonflement)",
        ]),

        ("mental_health_moderate", [
        
            r"(muy\s+triste|very\s+sad|très\s+triste).{0,30}(semanas?|weeks?|tiempo|time)",
            r"(deprimido|depressed|déprimé).{0,20}(semanas?|tiempo|weeks?)",
            r"(no\s+veo\s+sentido|don'?t\s+see\s+the\s+point|sans\s+but)",
            r"(tristeza|sadness|tristesse).{0,20}(no\s+se\s+va|won'?t\s+go\s+away|semanas?|weeks?)",
        ]),
    ]

    # ── SAFETY-SENSITIVE patterns ─────────────────────────────────────────────
    SAFETY_SENSITIVE_PATTERNS: List[str] = [
        r"(suicid|self.?harm|automutil)",
        r"(quiero|want|veux)\s+(morir|die|mourir|hacerme\s+daño)",
        r"(no\s+quiero\s+vivir|don'?t\s+want\s+to\s+live|ne\s+veux\s+plus\s+vivre)",
        r"(pastillas?\s+para\s+morir|pills?\s+to\s+die)",
        r"(muy\s+triste|very\s+sad|très\s+triste).{0,30}(semanas?|weeks?|tiempo)",
        r"(no\s+veo\s+sentido|don'?t\s+see\s+the\s+point|sans\s+but)",
        r"(deprimido|depressed|déprimé).{0,20}(semanas?|weeks?|tiempo)",
        r"(panic\s+attack|ataque\s+de\s+pánico|crise\s+de\s+panique)",
        r"(pensamientos|thoughts|pensées).{0,20}(suicid|muerte|morir)",
        r"(sobredosis|overdose|surdosage)",
        r"\d+\s*(ibuprofeno?|paracetamol|pastillas?|pills?).{0,20}(junto|together|ensemble|a\s+la\s+vez)",
        r"(mezclar|mix|mélanger).{0,20}(medicamentos|alcohol|drugs)",
        r"(no\s+como|not\s+eating|ne\s+mange\s+pas).{0,20}(días|days|jours)",
        r"(vomito|vomiting|vomis).{0,20}(a\s+propósito|intentionally|exprès)",
        r"(perder\s+peso|lose\s+weight|perdre\s+du\s+poids).{0,20}(rápido|fast|vite|pastillas|pills)",
        r"(me\s+pega|hits?\s+me|me\s+frappa|me\s+golpea)",
        r"(violencia|violence|abuse|maltrato)",
        r"(كنبغي\s*نموت|ما\s*بغيتش\s*نعيش|كنضرب\s*روحي)",
        r"(estoy|me\s+siento)\s+(un\s+poco|algo)\s+(triste|bajo|mal).{0,20}(hoy|today|aujourd'hui)",
        r"(i\s+feel|i'm)\s+(a\s+bit|kind\s+of)\s+(sad|down).{0,20}(today)",
    ]

    # ── LOW RISK ──────────────────────────────────────────────────────────────
    LOW_RISK_RULES: List[Tuple[str, List[str]]] = [

        ("wellness_question", [
            r"\b(cómo|how|comment)\b.{0,30}\b(dormir\s+mejor|sleep\s+better|mieux\s+dormir)\b",
            r"\b(cómo|how|comment)\b.{0,30}\b(reducir\s+estrés|reduce\s+stress|réduire\s+le\s+stress)\b",
            r"\b(consejos|tips|conseils)\b.{0,30}\b(salud|health|santé)\b",
            r"\b(qué\s+comer|what\s+to\s+eat|quoi\s+manger)\b",
            r"\b(ejercicio|exercise|exercice)\b.{0,30}\b(recomendado|recommended|recommandé)\b",
        ]),

        
        ("low_grade_fever", [
            r"(fiebre|fever|fièvre|temperatura|temperature).{0,20}(37[\.,]\d)",
            r"(37[\.,]\d).{0,20}(grados|degrees|°|fiebre|fever|temperatura)",
            r"(hijo|niño|bebé|child|baby|kid|enfant).{0,30}(37[\.,]\d)",
            r"(37[\.,]\d).{0,30}(hijo|niño|bebé|child|baby|kid|enfant)",
        ]),

        
        ("toilet_paper_blood", [
            r"sangre\s+(en\s+el\s+papel|al\s+limpiar|en\s+el\s+papel\s+higiénico)",
            r"blood\s+(on\s+the\s+paper|when\s+i\s+wipe|on\s+toilet\s+paper)",
            r"sang\s+(sur\s+le\s+papier|en\s+me\s+essuyant)",
            r"(papel|toilet\s+paper|papier).{0,20}(sangre|blood|sang)",
            r"(limpiarme|wipe|essuyer).{0,10}(sangre|blood|sang)",
        ]),

        ("minor_symptom", [
            r"(verruga|wart|verrue)",
            r"(ojo\s+rojo|red\s+eye|œil\s+rouge).{0,20}(?!.*(dolor\s+severo|severe\s+pain|douleur\s+sévère))",
            r"(picadura|bite|piqûre).{0,20}(insecto|insect|insecte).{0,20}(?!.*(alergia\s+severa|severe\s+allergy))",
            r"(resfrío|resfriado\s+leve|mild\s+cold|rhume\s+léger)",
            r"(tos\s+leve|mild\s+cough|toux\s+légère).{0,20}(?!.*días)",
        ]),

        ("information_request", [
            r"(qué\s+es|what\s+is|qu'?est.?ce\s+que).{0,30}(vitamina|vitamin|vitamine)",
            r"(qué\s+es|what\s+is|qu'?est.?ce\s+que).{0,30}(ibuprofeno|ibuprofen|paracetamol)",
            r"(información|information).{0,20}(sobre|about|sur)",
            r"(explica|explain|expliquer).{0,20}(qué|what|que)",
        ]),

        ("prescription_renewal", [
            r"(renovar|renew|renouveler).{0,20}(receta|prescription|ordonnance)",
            r"(receta|prescription|ordonnance).{0,20}(vencida|expired|expirée)",
        ]),
    ]



class RiskAssessmentAgent:
    """Risk Assessment Agent v3.2 — Manchester Triage System"""

    def __init__(self):
        self.logger = HealthChatLogger()
        self.rules = TriageRules()

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=10,
            api_key=os.getenv("GROQ_API_KEY")
        )
    

        self.assessment_prompt = PromptTemplate(
            input_variables=["question"],
            template=(
                "You are a medical triage assistant. Classify this health question.\n\n"
                "HIGH: severe symptoms, diagnosis requests, medication dosage, emergencies\n"
                "MEDIUM: specific symptoms lasting days, chronic condition questions, minor injuries\n"
                "LOW: general wellness info, mild symptoms, information requests\n\n"
                "Respond with ONE word only: HIGH, MEDIUM, or LOW\n\n"
                "Question: {question}\nRisk Level:"
            )
        )

    def assess(self, question: str, use_llm: bool = True) -> Dict[str, Any]:
        start_time = time.time()

        risk_level, rule_matched, category = self._assess_rules(question)
        method = "rules"

        if rule_matched is None and use_llm:
            risk_level = self._assess_llm(question)
            method = "llm_fallback"
            rule_matched = "llm"
            category = "general"

        is_medical_urgent    = risk_level in (RiskLevel.HIGH,)
        is_safety_sensitive  = self._check_safety_sensitive(question)

        confidence = self._calculate_confidence(question, rule_matched, method, is_safety_sensitive)
        reasoning  = self._generate_reasoning(risk_level, category, rule_matched)
        assessment_time = time.time() - start_time

        self.logger.log_metrics(
            "risk_assessment_time", assessment_time,
            {
                "risk_level":          risk_level.value,
                "is_medical_urgent":   is_medical_urgent,
                "is_safety_sensitive": is_safety_sensitive,
                "confidence":          confidence,
                "method":              method,
                "rule_matched":        rule_matched,
                "category":            category,
            }
        )

        return {
            "risk_level":          risk_level,
            "is_medical_urgent":   is_medical_urgent,
            "is_safety_sensitive": is_safety_sensitive,
            "confidence":          confidence,
            "reasoning":           reasoning,
            "assessment_time":     assessment_time,
            "method":              method,
            "rule_matched":        rule_matched,
            "category":            category,
        }

    def _assess_rules(self, question: str) -> Tuple[RiskLevel, str, str]:
        q = question.lower()

       
        for rule_name, patterns in self.rules.LOW_RISK_RULES:
            for pattern in patterns:
                if re.search(pattern, q, re.IGNORECASE):
                    return RiskLevel.LOW, rule_name, "low"

        # Then HIGH
        for rule_name, patterns in self.rules.HIGH_RISK_RULES:
            for pattern in patterns:
                if re.search(pattern, q, re.IGNORECASE):
                    return RiskLevel.HIGH, rule_name, "high"

        # Then MEDIUM
        for rule_name, patterns in self.rules.MEDIUM_RISK_RULES:
            for pattern in patterns:
                if re.search(pattern, q, re.IGNORECASE):
                    return RiskLevel.MEDIUM, rule_name, "medium"

        return RiskLevel.LOW, None, None

    def _assess_llm(self, question: str) -> RiskLevel:
        try:
            chain = self.assessment_prompt | self.llm
            response = chain.invoke({"question": question})
            result = response.content.strip().upper()
            if "HIGH" in result:
                return RiskLevel.HIGH
            elif "MEDIUM" in result:
                return RiskLevel.MEDIUM
            else:
                return RiskLevel.LOW
        except Exception as e:
            self.logger.log_error(error_type="risk_llm_error", error_message=str(e), question=question)
            return RiskLevel.LOW

    def _check_safety_sensitive(self, question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q, re.IGNORECASE) for p in self.rules.SAFETY_SENSITIVE_PATTERNS)

    def _calculate_confidence(self, question, rule_matched, method, is_safety_sensitive) -> float:
        if method == "llm_fallback":
            base = 0.55
        elif rule_matched:
            base = 0.85
        else:
            base = 0.50

        q = question.lower()
        specificity_boost = 0.0
        if re.search(r'\d+', q): specificity_boost += 0.04
        if re.search(r'(días|days|jours|semanas|weeks|horas|hours)', q): specificity_boost += 0.04
        if re.search(r'(severo|severe|intense|insoportable|unbearable|très fort|شديد)', q): specificity_boost += 0.03
        sensitivity_penalty = -0.03 if is_safety_sensitive else 0.0

        return round(min(0.98, max(0.40, base + specificity_boost + sensitivity_penalty)), 2)

    def _generate_reasoning(self, risk_level: RiskLevel, category: str, rule: str) -> str:
        reasoning_map = {
            "fever_high_child":             "High fever (38.5°C+) in a child — requires same-day medical evaluation.",
            "fever_very_high_adult":         "Very high or persistent fever (40°C+) — urgent evaluation needed.",
            "severe_abdominal_pain":         "Severe abdominal pain — may indicate appendicitis or other acute condition.",
            "neurological_symptoms":         "Sudden neurological symptoms — stroke/TIA must be ruled out.",
            "severe_allergic":               "Severe allergic reaction signs — anaphylaxis risk.",
            "high_fever_prolonged":          "High fever lasting 2-3 days without improvement — medical evaluation today.",
            "infection_signs":               "Signs of wound infection — antibiotic evaluation needed.",
            "difficulty_swallowing_severe":  "Significant swallowing difficulty — requires ENT or urgent evaluation.",
            "blood_symptoms_non_emergency":  "Blood in urine or dark stools — requires urgent workup.",
            "mental_health_crisis":          "Mental health crisis indicators — immediate support needed.",
            "unintentional_weight_loss":     "Unintentional weight loss — requires medical investigation.",
            "medication_dosage_request":     "Medication dosage request — safety filter will handle.",
            "persistent_fever_mild":         "Mild but persistent fever — see doctor this week.",
            "sore_throat_days":              "Sore throat lasting several days — may need throat swab or antibiotics.",
            "persistent_cough":              "Persistent cough — needs evaluation if lasting over 3 weeks.",
            "urinary_symptoms":              "Urinary symptoms — likely UTI, needs evaluation.",
            "skin_rash_days":                "Persistent skin rash — needs dermatological assessment.",
            "headache_frequent":             "Frequent headaches — needs evaluation to rule out underlying causes.",
            "digestive_persistent":          "Persistent digestive symptoms — GI evaluation recommended.",
            "chronic_condition_question":    "Chronic condition question — GP follow-up recommended.",
            "anxiety_sleep":                 "Sleep/anxiety issues — mental health support recommended.",
            "minor_injury":                  "Minor injury — may need clinical assessment.",
            "mental_health_moderate":        "Moderate mental health symptoms (weeks-long) — professional support recommended.",
            "wellness_question":             "General wellness question — educational response appropriate.",
            "low_grade_fever":               "Subfebril temperature (37.x°C) — monitor, no urgent action needed.",
            "toilet_paper_blood":            "Blood on toilet paper — likely hemorrhoids, low urgency.",
            "minor_symptom":                 "Minor symptom — self-care advice appropriate.",
            "information_request":           "Information request — educational response appropriate.",
            "prescription_renewal":          "Prescription renewal — routine GP appointment.",
            "llm":                           f"Classified by LLM as {risk_level.value} — no specific rule matched.",
        }

        if rule and rule in reasoning_map:
            return reasoning_map[rule]

        generic = {
            RiskLevel.HIGH:   "Symptoms or request indicate urgent medical evaluation needed.",
            RiskLevel.MEDIUM: "Specific symptoms mentioned — professional evaluation recommended.",
            RiskLevel.LOW:    "General health question — educational response appropriate.",
        }
        return generic[risk_level]


if __name__ == "__main__":
    agent = RiskAssessmentAgent()

    
    test_cases = [
        
        ("Tengo sangre en el papel",                         RiskLevel.LOW,    False, False),
        ("blood on toilet paper when I wipe",                RiskLevel.LOW,    False, False),
        ("Mi hijo tiene 37.5 de fiebre",                     RiskLevel.LOW,    False, False),
        ("My child has a temperature of 37.5",               RiskLevel.LOW,    False, False),
        ("Estoy un poco triste hoy",                         RiskLevel.LOW,    False, True),
        ("I'm feeling a bit sad today",                      RiskLevel.LOW,    False, True),
        ("No veo sentido a nada",                            RiskLevel.MEDIUM, False, True),
        ("Quiero hacerme daño",                              RiskLevel.HIGH,   True,  True),
        ("Mi hijo tiene 39 de fiebre",                       RiskLevel.HIGH,   True,  False),
        ("Sangre en la orina",                               RiskLevel.HIGH,   True,  False),
        ("Heces negras desde ayer",                          RiskLevel.HIGH,   True,  False),
        ("Me siento muy triste desde hace semanas",          RiskLevel.MEDIUM, False, True),
        ("Estoy deprimida desde hace semanas",               RiskLevel.MEDIUM, False, True),
        ("Llevo 3 días con dolor de garganta",               RiskLevel.MEDIUM, False, False),
        ("Tengo una verruga en el pie",                      RiskLevel.LOW,    False, False),
    ]

    print("=" * 80)
    print("RISK AGENT v3.2 — REGRESSION + FIX TESTS")
    print("=" * 80)

    passed = 0
    for q, er, eu, es in test_cases:
        r = agent.assess(q, use_llm=False)
        ok = r['risk_level'] == er and r['is_medical_urgent'] == eu and r['is_safety_sensitive'] == es
        passed += ok
        status = "✅" if ok else "❌"
        print(
            f"{status} [{r['risk_level'].value:<6}] "
            f"urg={int(r['is_medical_urgent'])} sens={int(r['is_safety_sensitive'])} "
            f"rule={str(r['rule_matched']):<28} | {q}"
        )

    print(f"\n{'='*80}")
    print(f"PASSED: {passed}/{len(test_cases)} — Accuracy: {passed/len(test_cases)*100:.1f}%")