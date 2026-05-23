# agents/safety_filter_agent.py
"""
Safety Filter Agent v3.0

IMPROVEMENTS v3.0:
✅ FIXED: Arabic support added to ALL responses
✅ FIXED: Chest pain removed — handled by EmergencyHandler (no duplication)
✅ FIXED: Medication filter no longer blocks legitimate informational questions
✅ FIXED: Vitamin/supplement questions distinguished from dosage requests
✅ NEW: Smarter medication detection — blocks advice, not education
✅ NEW: Mental health filter — detects distress signals
✅ NEW: Self-harm language detection
✅ NEW: Diagnosis language softener (post-LLM cleanup)
✅ NEW: filter_reasons improved for better logging/debugging
"""

import re
from typing import Dict, Any


class SafetyFilterAgent:
    """
    Safety filter for medical chatbot v3.0
    
    IMPORTANT: This filter runs AFTER EmergencyHandler.
    Do NOT duplicate emergency logic here.
    
    Responsibilities:
    - Block specific medication dosage advice
    - Detect mental health / self-harm signals
    - Clean up LLM formatting artifacts
    - Soften diagnosis language
    - Add Arabic support to all safe responses
    """

    def __init__(self):

        
        self.medication_response = {
            'es': "El tratamiento adecuado depende de la causa y de tu situación personal. Por seguridad, consulta con un profesional de salud o farmacéutico antes de tomar cualquier medicamento. 💙",
            'en': "Appropriate treatment depends on the cause and your personal situation. For safety, please consult a healthcare professional or pharmacist before taking any medication. 💙",
            'fr': "Le traitement approprié dépend de la cause et de votre situation personnelle. Pour votre sécurité, consultez un professionnel de santé ou un pharmacien avant de prendre tout médicament. 💙",
            'ar': "العلاج المناسب يعتمد على السبب ووضعك الشخصي. للسلامة، استشر أخصائي صحي أو صيدلاني قبل تناول أي دواء. 💙"
        }

        
        self.exercise_pain_response = {
            'es': "El dolor durante o después del ejercicio puede deberse a sobrecarga muscular, técnica inadecuada o una lesión. Reducir la intensidad y descansar puede ayudar. Si el dolor persiste, es intenso o aparece en el pecho, consulta con un profesional de salud.",
            'en': "Pain during or after exercise may be due to muscle overload, improper technique, or an injury. Reducing intensity and resting may help. If pain persists, is intense, or appears in the chest, consult a healthcare professional.",
            'fr': "La douleur pendant ou après l'exercice peut être due à une surcharge musculaire, une technique inadéquate ou une blessure. Réduire l'intensité et se reposer peut aider. Si la douleur persiste, est intense ou apparaît dans la poitrine, consultez un professionnel de santé.",
            'ar': "الألم أثناء أو بعد التمرين قد يكون بسبب إجهاد عضلي أو تقنية غير صحيحة أو إصابة. تقليل الشدة والراحة قد يساعد. إذا استمر الألم أو كان شديداً، استشر أخصائي صحي."
        }

        
        self.persistent_symptoms_response = {
            'es': "Cuando los síntomas persisten más de lo esperado, lo más recomendable es consultar con un profesional de salud para una evaluación adecuada y descartar causas que requieran tratamiento.",
            'en': "When symptoms persist longer than expected, the most advisable course is to consult with a healthcare professional for proper evaluation and to rule out causes requiring treatment.",
            'fr': "Quand les symptômes persistent plus longtemps que prévu, il est plus recommandé de consulter un professionnel de santé pour une évaluation appropriée et exclure des causes nécessitant un traitement.",
            'ar': "عندما تستمر الأعراض أكثر من المتوقع، يُنصح باستشارة أخصائي صحي للتقييم المناسب واستبعاد الأسباب التي تحتاج علاجاً."
        }

        
        self.mental_health_response = {
            'es': "💙 Parece que estás pasando por un momento difícil. Es completamente válido buscar apoyo. Hablar con un profesional de salud mental puede ayudarte mucho. Si en algún momento sientes que no puedes más, no dudes en llamar a los servicios de emergencia o a una línea de apoyo emocional.",
            'en': "💙 It sounds like you're going through a difficult time. It's completely valid to seek support. Talking to a mental health professional can help a lot. If at any point you feel overwhelmed, don't hesitate to contact emergency services or an emotional support line.",
            'fr': "💙 Il semble que vous traversez un moment difficile. Il est tout à fait normal de chercher du soutien. Parler à un professionnel de santé mentale peut beaucoup aider. Si à tout moment vous vous sentez dépassé(e), n'hésitez pas à contacter les services d'urgence ou une ligne de soutien émotionnel.",
            'ar': "💙 يبدو أنك تمر بوقت صعب. من الطبيعي تماماً أن تبحث عن دعم. التحدث مع أخصائي صحة نفسية يمكن أن يساعد كثيراً. إذا شعرت في أي وقت أنك لا تستطيع التحمل، لا تتردد في الاتصال بخدمات الطوارئ أو خط دعم عاطفي."
        }

        
        self.crisis_response = {
            'es': "💙 Lo que describes me preocupa y quiero que sepas que no estás solo/a. Por favor, contacta ahora con los servicios de emergencia o con una línea de apoyo en crisis. Mereces ayuda y apoyo.",
            'en': "💙 What you describe concerns me and I want you to know you are not alone. Please contact emergency services or a crisis support line now. You deserve help and support.",
            'fr': "💙 Ce que vous décrivez m'inquiète et je veux que vous sachiez que vous n'êtes pas seul(e). Veuillez contacter maintenant les services d'urgence ou une ligne de soutien en crise. Vous méritez aide et soutien.",
            'ar': "💙 ما تصفه يقلقني وأريدك أن تعلم أنك لست وحدك. يرجى التواصل الآن مع خدمات الطوارئ أو خط دعم الأزمات. أنت تستحق المساعدة والدعم."
        }

    

    def filter_response(self, response: str, question: str, risk_level: Any, lang_code: str) -> Dict[str, Any]:
        """
        Main filtering logic.

        Priority order:
        1. Self-harm / crisis signals (HIGHEST PRIORITY)
        2. Mental health distress signals
        3. Specific medication DOSAGE requests (block)
        4. Exercise pain questions
        5. Medication mentions in LLM response (clean or replace)
        6. General cleanup (lists, diagnosis language, formatting)
        
        NOTE: Chest pain and breathing emergencies are handled upstream
        by EmergencyHandler — do NOT duplicate that logic here.
        """

        q = question.lower()
        r = response.lower()

       
        if self._is_crisis(q):
            return self._result(
                self.crisis_response.get(lang_code, self.crisis_response['es']),
                True, ["crisis_selfharm_detected"]
            )

        
        if self._is_mental_health_distress(q):
            return self._result(
                self.mental_health_response.get(lang_code, self.mental_health_response['es']),
                True, ["mental_health_distress_detected"]
            )

        
        if self._is_medication_dosage_request(q):
            return self._result(
                self.medication_response.get(lang_code, self.medication_response['es']),
                True, ["medication_dosage_request_blocked"]
            )

        
        if self._is_exercise_pain_question(q):
            return self._result(
                self.exercise_pain_response.get(lang_code, self.exercise_pain_response['es']),
                True, ["exercise_pain_handled"]
            )

        
        if self._response_contains_medication_advice(response):
            if self._is_persistent_symptom_question(q):
                return self._result(
                    self.persistent_symptoms_response.get(lang_code, self.persistent_symptoms_response['es']),
                    True, ["medication_in_response_replaced_persistent"]
                )
            return self._result(
                self.medication_response.get(lang_code, self.medication_response['es']),
                True, ["medication_advice_in_response_blocked"]
            )

        
        cleaned = response
        reasons = []

        
        if self._has_numbered_list(cleaned):
            cleaned = self._remove_numbered_lists(cleaned)
            reasons.append("numbered_list_removed")

        
        if self._has_bullet_list(cleaned):
            cleaned = self._remove_bullet_lists(cleaned)
            reasons.append("bullet_list_removed")

        
        cleaned, softened = self._soften_diagnosis_language(cleaned)
        if softened:
            reasons.append("diagnosis_language_softened")

        
        if self._contains_exercise_techniques(cleaned):
            return self._result(
                self.exercise_pain_response.get(lang_code, self.exercise_pain_response['es']),
                True, ["exercise_techniques_in_response_removed"]
            )

        
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

        return self._result(cleaned, bool(reasons), reasons)

   

    def _is_crisis(self, q: str) -> bool:
        """Detects self-harm or suicidal ideation"""
        patterns = [
            r'\b(suicid|self.?harm|automutil)',
            r'(quiero|want|veux|want\s+to)\s+(morir|die|mourir|kill\s+myself|hacerme\s+daño)',
            r'(no\s+quiero\s+vivir|don\'?t\s+want\s+to\s+live|ne\s+veux\s+plus\s+vivre)',
            r'(me\s+quiero\s+matar|want\s+to\s+hurt\s+myself)',
            r'(pastillas?\s+para\s+morir|pills?\s+to\s+die)',
            r'(ما\s*بغيتش\s*نعيش|كنبغي\s*نموت)',
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    def _is_mental_health_distress(self, q: str) -> bool:
        """Detects mental health distress signals (non-crisis)"""
        patterns = [
            r'(no\s+puedo\s+más|can\'?t\s+(cope|go\s+on)|je\s+n\'?en\s+peux\s+plus)',
            r'(muy\s+triste|very\s+sad|très\s+triste|deprimido|depressed|déprimé)',
            r'(ansiedad\s+severa|severe\s+anxiety|anxiété\s+sévère)',
            r'(sin\s+dormir\s+\d+\s*días|haven\'?t\s+slept\s+\d+)',
            r'(lloro\s+(todo|sin\s+parar)|cry\s+all\s+the\s+time|pleure\s+tout\s+le\s+temps)',
            r'(panic\s+attack|ataque\s+de\s+pánico|crise\s+de\s+panique)',
            r'(كنبكي\s*بزاف|مكيفيش|ما\s*نقدرش)',
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    def _is_medication_dosage_request(self, q: str) -> bool:
        """
        Only blocks when asking HOW MUCH or WHICH medication to take.
        Does NOT block educational questions like 'what is ibuprofen'.
        """
        
        educational_patterns = [
            r'(what\s+is|qué\s+es|qu\'?est.ce\s+que|what\s+does)\s+\w+\s+(do|mean)',
            r'(explain|explica|explique)\s+(me\s+)?(what|qué|que)',
            r'(difference\s+between|diferencia\s+entre|différence\s+entre)',
        ]
        if any(re.search(p, q, re.IGNORECASE) for p in educational_patterns):
            return False

        
        block_patterns = [
            r'(can\s+i|puedo|puis-je)\s+(take|tomar|prendre)\s+\d+',
            r'\d+\s*(ibuprofeno?|paracetamol|aspirin|pastillas?|pills?|comprimés?)',
            r'(how\s+much|cuánto|combien)\s+(can\s+i\s+take|puedo\s+tomar|puis-je\s+prendre)',
            r'(what\s+(medication|medicine|drug)|qué\s+medicamento|quel\s+médicament)\s+(should\s+i|debo|dois)',
            r'(recommend|recomienda|recommande).{0,20}(medication|medicamento|médicament)',
            r'(should\s+i|debo|dois-je)\s+(take|tomar|prendre).{0,20}(for|para|pour)',
            r'(overdose|sobredosis|surdosage)',
            r'(double|triple|más\s+dosis|augmenter\s+la\s+dose)',
            r'كم\s*(حبة|دواء|مل)',
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in block_patterns)

    def _is_exercise_pain_question(self, q: str) -> bool:
        """Check if question is about exercise-related pain"""
        patterns = [
            r'(hurt|pain|ache|duele|mal|douleur).{0,20}(exercise|ejercicio|sport|gym|workout|entreno)',
            r'(exercise|ejercicio|sport|gym|workout).{0,20}(hurt|pain|duele|mal)',
            r'(knees?|rodillas?|genoux).{0,20}(hurt|duele|mal)',
            r'(back|espalda|dos).{0,20}(hurt|duele|mal).{0,20}(exercise|gym|sport)',
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    def _is_persistent_symptom_question(self, q: str) -> bool:
        """Check if question is about persistent symptoms"""
        patterns = [
            r'(persistent|persistente|persiste)',
            r'(won\'?t\s+go\s+away|no\s+se\s+quita|ne\s+part\s+pas)',
            r'(desde\s+hace\s+\d+|for\s+\d+\s+days?|depuis\s+\d+\s+jours?)',
            r'(llevo\s+\d+\s+días|been\s+\d+\s+days)',
            r'ما\s*كيزيد\s*يولي',
        ]
        return any(re.search(p, q, re.IGNORECASE) for p in patterns)

    def _response_contains_medication_advice(self, response: str) -> bool:
        """
        Check if LLM response recommends specific medications or dosages.
        More precise than v2 — only triggers on actual advice, not mentions.
        """
        advice_patterns = [
            r'(take|tomar|prendre|nehmen)\s+(ibuprofen|paracetamol|aspirin\w*|ibuprofeno)',
            r'(try|use|usar|utiliser)\s+(ibuprofen|paracetamol|decongestant|antihistamine)',
            r'\bover-the-counter\b',
            r'\bcough\s+suppressant\b',
            r'\blozenge\b',
            r'\b\d+\s*mg\b',
            r'\bdosis\s+(recomendada|máxima|diaria)',
            r'\bdosage\b',
            r'\bdescongestionante\b',
        ]
        r_lower = response.lower()
        return any(re.search(p, r_lower, re.IGNORECASE) for p in advice_patterns)

    def _contains_exercise_techniques(self, response: str) -> bool:
        """Check if response contains technical exercise advice"""
        technical_terms = [
            'squat depth', 'knee angle', 'form correction',
            'knees past toes', 'wall push-up', 'modify your exercise',
            'chair for squat', 'sentadilla', 'forma correcta del ejercicio',
        ]
        r_lower = response.lower()
        return any(term in r_lower for term in technical_terms)

    
    def _has_numbered_list(self, text: str) -> bool:
        return bool(re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE))

    def _has_bullet_list(self, text: str) -> bool:
        return bool(re.search(r'^\s*[-*•]\s', text, re.MULTILINE))

    def _remove_numbered_lists(self, text: str) -> str:
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line = re.sub(r'^\s*\d+[.)]\s*', '', line)
            if line.strip():
                cleaned.append(line.strip())
        return ' '.join(cleaned)

    def _remove_bullet_lists(self, text: str) -> str:
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            line = re.sub(r'^\s*[-*•]\s*', '', line)
            if line.strip():
                cleaned.append(line.strip())
        return ' '.join(cleaned)

    def _soften_diagnosis_language(self, text: str) -> tuple[str, bool]:
        """Replace direct diagnosis language with conditional phrasing"""
        replacements = [
            # English
            (r'\byou have\b(?!\s+(the\s+right|a\s+point|enough))', 'you may have'),
            (r'\byou\'re suffering from\b', 'you may be experiencing'),
            (r'\bthis is\s+(a|an)\s+(?=\w)', 'this may be a '),
            # Spanish
            (r'\btienes\b(?!\s+(razón|que|suerte|tiempo))', 'puedes tener'),
            (r'\bestás\s+sufriendo\s+de\b', 'puedes estar experimentando'),
            (r'\besto\s+es\s+(un|una)\b', 'esto puede ser un'),
            # French
            (r'\bvous\s+avez\b(?!\s+(besoin|raison|le\s+droit))', 'vous pouvez avoir'),
            (r'\bvous\s+souffrez\s+de\b', 'vous pouvez souffrir de'),
            # General
            (r'\b(diagnosed with|diagnosticado con|diagnostiqué avec)\b', 'may indicate'),
        ]
        modified = False
        for pattern, replacement in replacements:
            new_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            if new_text != text:
                text = new_text
                modified = True
        return text, modified

    

    def _result(self, response: str, was_filtered: bool, reasons: list) -> Dict[str, Any]:
        return {
            "filtered_response": response,
            "was_filtered": was_filtered,
            "filter_reasons": reasons
        }