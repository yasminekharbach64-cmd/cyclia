"""
PROFESSIONAL VERSION v3.12 - SPECIALIZED IN FEMALE HORMONAL HEALTH
==================================================================

CHANGES v3.8 (over v3.7) — Alicia's feedback + dataset coverage:

✅ FIX 1: System preamble enriched with Alicia's exact domain description
         (covers metabolism, full female lifespan hormonal changes,
         tiroid health linked to cycle phases, etc.)

✅ FIX 2: New anti-pattern rules in system preamble:
         - Spanish quality: gender/number agreement, correct accents
           (prevents "las niveles", "Exercicio", "trigliceridos" errors)
         - No literal interpretation of metaphorical/emotional questions
           (e.g. "voy a ciegas" should not be read as cognitive impairment)
         - No automatic "consulta ginecólogo" closing on every response —
           recommend professional ONLY when clinically warranted
         - Vary closings to avoid repetitive "cuídate" / "espero que te
           encuentres mejor pronto"

✅ FIX 3: low_risk_template extended with 5 new few-shot examples
         covering Alicia's dataset blocks 8, 9, 10, 11:
         - Meta-IA ("¿hasta dónde puede ayudar una IA?")
         - Metaphorical/emotional ("¿voy a ciegas?")
         - Cultural ("¿por qué las famosas no tienen menopausia?")
         - Cognitive fog ("¿es normal olvidar palabras?")
         - Sexual (low libido)

✅ FIX 4: mental_health_template extended with 2 emotional-validation
         examples that don't pathologize:
         - "¿Estoy exagerando o realmente me está pasando algo?"
         - "¿Por qué siento que no soy la misma?"

✅ FIX 5: AI meta-responses refined — slightly warmer, less robotic,
         consistent voice across general/cuando_medico/puede_recomendar.

✅ FIX 6: Closing phrases pool expanded (was 2 per language, now 5)
         to reduce repetition across consecutive answers.

✅ FIX 7: Knowledge base size updated in docstring (484 → 2527 papers,
         reflecting the expanded PubMed corpus + semantic search v3.0).

CHANGES v3.7 (over v3.6) — Honest fallback when no PubMed evidence:

✅ FIX 1: When the RAG agent doesn't find relevant PubMed papers
         (low confidence), the system now adds an HONEST DISCLAIMER
         instead of letting the LLM fabricate medical claims.

✅ FIX 2: Stricter LLM prompt when there's no high-confidence context.

✅ FIX 3: Works in tandem with rag_agent.py v3.0 (semantic search,
         multilingual embeddings, 2527 papers covering the full female
         hormonal health spectrum).

CHANGES v3.6 (over v3.5):
✅ System prompt translated to Spanish to prevent English vocabulary leak.
✅ Anti-hallucination instructions in system prompt.
✅ Dedicated handler for AI/app meta-questions (Block 8 of dataset).
✅ Length guard on responses.

CHANGES v3.5 (over v3.4):
✅ "¿Qué tomo para la gripe?" correctly redirects out-of-scope.
✅ Works in tandem with rag_agent.py v2.0 (PubMed citations).

CHANGES v3.4 (over v3.3) — Following supervisor (Alicia) feedback:
✅ System SPECIALIZED in female hormonal health.
✅ Out-of-scope detection.
✅ All prompt templates rewritten with female hormonal health examples.
✅ Greeting message reflects specialization.

KNOWLEDGE BASE: 2527 PubMed papers on female hormonal health
                (curated by pubmed_fetcher.py v2.0, indexed with
                semantic embeddings via sentence-transformers v3.0)
"""
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
load_dotenv()
from langchain_core.prompts import PromptTemplate
from logger import HealthChatLogger
from agents.risk_agent import RiskAssessmentAgent, RiskLevel
from agents.rag_agent import RAGAgent
from agents.knowledge_base import MEDICAL_KNOWLEDGE_BASE
from agents.response_normalizer import ResponseNormalizer
from agents.emergency_handler_agent import EmergencyHandler
from conversation_memory import ConversationMemory
import time
import re
import random


_agent_instance = None


def get_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = IntegratedHealthAgent()
    return _agent_instance


class IntegratedHealthAgent:
    """Professional Specialized Health Triage Agent v3.12

    Domain: Female hormonal health across the full lifespan
    Audience: Women across all life stages (reproductive age, perimenopause,
              menopause, postmenopause)
    """

    # ─────────────────────────────────────────────────────────────────────
    # SCOPE DEFINITION — Female hormonal health domain
    # ─────────────────────────────────────────────────────────────────────

    # Keywords that indicate the question IS within scope
    IN_SCOPE_KEYWORDS = {
        # Menstrual cycle
        'es': [
            'menstruac', 'regla', 'ciclo', 'período', 'periodo', 'sangrado',
            'menstrual', 'ovulac', 'fase lútea', 'fase folicular', 'amenorrea',
            'dismenorrea', 'menorragia', 'spotting', 'manchado',
            # Fertility / reproductive
            'fertilidad', 'embaraz', 'concepc', 'ovari', 'útero', 'utero',
            'endometri', 'mioma', 'fibroma', 'sop', 'pcos', 'poliquíst',
            'quiste ovárico', 'trompas', 'salpinx',
            # Menopause
            'menopaus', 'perimenopaus', 'postmenopaus', 'climaterio',
            'sofoco', 'sofocos', 'bochorno', 'sudoración nocturna',
            # Hormones
            'hormona', 'hormonal', 'estrógeno', 'estrogeno', 'progesterona',
            'testosterona', 'fsh', 'lh', 'prolactina', 'tiroid', 'tsh', 't3', 't4',
            'hashimoto', 'hipotiroid', 'hipertiroid',
            # Hormone-related symptoms
            'síndrome premenstrual', 'sindrome premenstrual', 'spm', 'tdpm',
            'libido', 'sequedad vaginal', 'incontinencia',
            # Conditions
            'osteoporosis', 'densidad ósea', 'salud ósea',
            # Female-specific
            'mujer', 'mujeres', 'femenin', 'pecho', 'mama', 'mamas', 'pezón',
            'vagina', 'vulva', 'cervi',
            # Postpartum
            'posparto', 'postparto', 'lactancia', 'puerperio',
        ],
        'en': [
            'menstruat', 'period', 'cycle', 'bleeding', 'ovulat',
            'luteal', 'follicular', 'amenorrhea', 'dysmenorrhea', 'menorrhagia',
            'spotting',
            'fertility', 'pregnan', 'concept', 'ovar', 'uterus', 'uterine',
            'endometri', 'fibroid', 'pcos', 'polycystic',
            'menopaus', 'perimenopaus', 'postmenopaus', 'climacteric',
            'hot flash', 'hot flush', 'night sweat',
            'hormone', 'hormonal', 'estrogen', 'oestrogen', 'progesterone',
            'testosterone', 'fsh', 'lh', 'prolactin', 'thyroid', 'tsh',
            'hashimoto', 'hypothyroid', 'hyperthyroid',
            'pms', 'pmdd', 'premenstrual',
            'libido', 'vaginal dryness', 'incontinence',
            'osteoporosis', 'bone density', 'bone health',
            'woman', 'women', 'female', 'breast', 'nipple',
            'vagina', 'vulva', 'cervi',
            'postpartum', 'breastfeeding', 'lactation',
        ],
        'fr': [
            'menstruat', 'règles', 'regles', 'cycle', 'saignement', 'ovulat',
            'lutéale', 'folliculaire', 'aménorrhée', 'dysménorrhée',
            'fertilité', 'grossesse', 'concept', 'ovair', 'utérus', 'uterus',
            'endométri', 'fibrome', 'sopk', 'polykyst',
            'ménopause', 'menopause', 'périménopause', 'postménopause',
            'bouffée de chaleur', 'sueur nocturne',
            'hormone', 'hormonal', 'œstrogène', 'estrogène', 'progestérone',
            'testostérone', 'thyroïd', 'thyroide',
            'syndrome prémenstruel', 'spm',
            'libido', 'sécheresse vaginale',
            'ostéoporose', 'densité osseuse',
            'femme', 'féminin', 'sein', 'mamelon',
            'vagin', 'vulve', 'col',
            'post-partum', 'allaitement',
        ],
        'ar': [
            'دورة', 'حيض', 'طمث', 'نزيف', 'إباضة',
            'خصوبة', 'حمل', 'مبيض', 'رحم',
            'انقطاع الطمث', 'سن اليأس',
            'هرمون', 'إستروجين', 'بروجسترون', 'غدة درقية',
            'متلازمة',
            'ثدي', 'مهبل',
            'نفاس', 'رضاعة',
        ],
    }

    def __init__(self):
        self.logger = HealthChatLogger()
        self.risk_agent = RiskAssessmentAgent()
        self.rag_agent = RAGAgent(MEDICAL_KNOWLEDGE_BASE)
        self.normalizer = ResponseNormalizer()
        self.memory = ConversationMemory()

        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            max_tokens=500,
            stop_sequences=["Q:", "---", "\n\n\n"],
            api_key=os.getenv("GROQ_API_KEY")
        )


        self._setup_varied_phrases()
        self._setup_out_of_scope_responses()
        self._setup_templates()

    def _setup_varied_phrases(self):
        # ✨ v3.8: expanded closing pool (was 2 per language) to reduce
        # repetition across consecutive answers in the same conversation.
        self.closings = {
            'es': [
                "Cuídate.",
                "Espero que te ayude.",
                "Un abrazo.",
                "Estoy aquí si quieres seguir hablando del tema.",
                "Ojalá esto te dé un poco de claridad.",
            ],
            'en': [
                "Take care.",
                "Hope this helps.",
                "Sending you a hug.",
                "I'm here if you want to keep exploring this.",
                "I hope this brings you a bit of clarity.",
            ],
            'fr': [
                "Prenez soin de vous.",
                "J'espère que cela vous aide.",
                "Je vous envoie un câlin.",
                "Je suis là si vous voulez continuer à en parler.",
                "J'espère que cela vous apporte un peu de clarté.",
            ],
            'ar': [
                "اعتن بنفسك.",
                "آمل أن يساعدك ذلك.",
                "أرسل لك عناقًا.",
                "أنا هنا إذا أردت متابعة الحديث."
            ]
        }

        self.vague_symptoms = {
            'es': [r'\bme\s+(?:encuentro|siento)\s+mal\b', r'\bno\s+me\s+encuentro\s+bien\b',
                   r'\bme\s+duele\b(?!\s+\w)', r'\btengo\s+dolor\b(?!\s+de)'],
            'en': [r"\bi\s+(?:don'?t\s+feel\s+well|feel\s+(?:bad|sick|unwell))\b",
                   r'\bi\s+have\s+pain\b(?!\s+in)', r'\bi\s+hurt\b'],
            'fr': [r'\bje\s+ne\s+me\s+sens\s+pas\s+bien\b', r"\bje\s+me\s+sens\s+mal\b",
                   r"\bj'?ai\s+mal\b(?!\s+(?:à|au|aux))"],
            'ar': [r'مش\s*مرتاح', r'تعبان', r'عندي\s*ألم']
        }

    # ─────────────────────────────────────────────────────────────────────
    # OUT-OF-SCOPE RESPONSES
    # ─────────────────────────────────────────────────────────────────────
    def _setup_out_of_scope_responses(self):
        """Polite redirection messages when the question is outside the
        scope of female hormonal health."""
        self.out_of_scope_responses = {
            'es': (
                "Mi especialidad es la salud hormonal femenina: ciclo menstrual, "
                "fertilidad, perimenopausia, menopausia, salud tiroidea en mujeres "
                "y los síntomas relacionados (sueño, estado de ánimo, metabolismo, etc.).\n\n"
                "Para esa consulta lo más adecuado es acudir a tu médico de cabecera. "
                "¿Hay algo sobre salud hormonal o ciclo femenino en lo que pueda ayudarte?"
            ),
            'en': (
                "My specialty is female hormonal health: menstrual cycle, fertility, "
                "perimenopause, menopause, thyroid health in women, and related "
                "symptoms (sleep, mood, metabolism, etc.).\n\n"
                "For that question, the best is to consult your general practitioner. "
                "Is there anything about hormonal or female cycle health I can help you with?"
            ),
            'fr': (
                "Ma spécialité est la santé hormonale féminine : cycle menstruel, "
                "fertilité, périménopause, ménopause, santé thyroïdienne chez la femme "
                "et les symptômes associés (sommeil, humeur, métabolisme, etc.).\n\n"
                "Pour cette question, le mieux est de consulter votre médecin généraliste. "
                "Y a-t-il quelque chose concernant la santé hormonale ou le cycle féminin "
                "avec lequel je peux vous aider ?"
            ),
            'ar': (
                "تخصصي هو الصحة الهرمونية للمرأة: الدورة الشهرية، الخصوبة، "
                "ما قبل انقطاع الطمث، انقطاع الطمث، صحة الغدة الدرقية لدى النساء، "
                "والأعراض المرتبطة بها.\n\n"
                "لهذا السؤال، الأفضل استشارة طبيبك العام. "
                "هل هناك شيء حول الصحة الهرمونية يمكنني مساعدتك فيه؟"
            ),
        }

        # ─────────────────────────────────────────────────────────────────
        # AI META-QUESTIONS (Block 8 from Alicia's dataset)
        # ─────────────────────────────────────────────────────────────────
        # Curated answers for questions about the chatbot's own role,
        # limits and proper use. v3.8: warmer, less robotic tone.
        self.ai_meta_responses = {
            'es': {
                'general': (
                    "Una asistente como yo puede ofrecerte información basada en evidencia "
                    "científica (en mi caso, papers de PubMed sobre salud hormonal femenina), "
                    "ayudarte a entender tus síntomas, prepararte mejor para una consulta "
                    "médica y orientarte sobre cuándo es importante acudir al especialista.\n\n"
                    "Lo que NO puedo hacer: diagnosticarte, recetarte tratamientos, sustituir "
                    "la exploración física de un profesional, ni interpretar tus análisis "
                    "sin un médico que conozca tu historial completo.\n\n"
                    "La forma más útil de usarme es como una herramienta complementaria: "
                    "para informarte, organizar tus dudas y entender mejor lo que te pasa, "
                    "validando siempre las decisiones importantes con tu ginecólogo o médico "
                    "de confianza."
                ),
                'cuando_medico': (
                    "Es momento de acudir a un profesional de la salud cuando los síntomas "
                    "son intensos o persistentes, cuando aparecen señales de alarma "
                    "(sangrado postmenopáusico, dolor pélvico fuerte, bultos), cuando tienes "
                    "dudas sobre medicación o tratamientos, o cuando simplemente necesitas "
                    "que alguien explore tu situación en persona.\n\n"
                    "Yo puedo ayudarte a organizar lo que quieres consultar y a entender "
                    "la información, pero la valoración clínica solo puede hacerla un médico."
                ),
                'puede_recomendar': (
                    "No, una app no debería decirte qué medicamento tomar. Esa decisión "
                    "corresponde a un profesional sanitario que conoce tu historial, "
                    "alergias y otras condiciones. Yo puedo darte información general "
                    "sobre los tipos de tratamientos disponibles y ayudarte a entender "
                    "lo que tu médico te explique, pero no sustituyo la prescripción médica."
                ),
            },
            'en': {
                'general': (
                    "An assistant like me can offer you information based on scientific "
                    "evidence (in my case, PubMed papers on female hormonal health), help "
                    "you understand your symptoms, prepare for a medical consultation, and "
                    "guide you on when it's important to see a specialist.\n\n"
                    "What I CANNOT do: diagnose you, prescribe treatments, replace a physical "
                    "examination, or interpret your test results without a doctor who knows "
                    "your full medical history.\n\n"
                    "The most useful way to work with me is as a complementary tool — for "
                    "information, organizing your questions, and understanding what's "
                    "happening — always validating important decisions with your gynecologist "
                    "or trusted doctor."
                ),
                'cuando_medico': (
                    "It's time to see a healthcare professional when symptoms are intense "
                    "or persistent, when warning signs appear (postmenopausal bleeding, "
                    "severe pelvic pain, lumps), when you have questions about medication, "
                    "or when you simply need someone to examine you in person.\n\n"
                    "I can help you organize your questions and understand information, "
                    "but clinical evaluation can only be done by a doctor."
                ),
                'puede_recomendar': (
                    "No, an app shouldn't tell you which medication to take. That decision "
                    "belongs to a healthcare professional who knows your history, allergies, "
                    "and other conditions. I can give you general information about available "
                    "treatments and help you understand what your doctor explains, but I do "
                    "not replace medical prescription."
                ),
            },
        }

    def _is_ai_meta_question(self, question: str, lang_code: str) -> str:
        """
        Detects questions about the AI/app itself (Block 8 of Alicia's dataset).
        Returns the response key ('general', 'cuando_medico', 'puede_recomendar')
        or None if not a meta-question.
        """
        q = question.lower()

        # "Can an app tell me what to take?" / "Should I take...?"
        recommendation_patterns = [
            r'(puede|can)\s+(una?\s+)?(app|ia|chatbot|inteligencia\s+artificial|ai)\s+(decirme|tell\s+me)',
            r'(app|ia|chatbot|ai)\s+(decirme|tell\s+me)\s+(qu[eé]\s+tomar|what\s+to\s+take)',
            r'qu[eé]\s+tipo\s+de\s+recomendaciones\s+son\s+seguras',
            r'recetar(me)?\s+(con|usando|via|por)\s+(app|ia|chatbot)',
        ]
        if any(re.search(p, q) for p in recommendation_patterns):
            return 'puede_recomendar'

        # "When should I go to a doctor?"
        doctor_patterns = [
            r'cu[áa]ndo\s+(tengo\s+que|debo|deber[ií]a)\s+ir\s+a',
            r'when\s+(should\s+i|do\s+i\s+have\s+to)\s+go',
            r'cu[áa]ndo.{0,30}(profesional|m[eé]dico|doctor|especialista)',
            r'c[oó]mo\s+saber\s+cu[áa]ndo.{0,30}(profesional|m[eé]dico|doctor)',
        ]
        if any(re.search(p, q) for p in doctor_patterns):
            return 'cuando_medico'

        # General AI/app questions
        general_patterns = [
            r'(hasta\s+d[oó]nde|how\s+far)\s+(puede|can)\s+(ayudar|help).{0,20}(ia|ai|app)',
            r'qu[eé]\s+(puede|can)\s+(hacer|do)\s+(una?\s+)?(ia|ai|app)',
            r'qu[eé]\s+puede\s+hacer\s+una?\s+(ia|ai|chatbot)\s+y\s+qu[eé]\s+no',
            r'(c[oó]mo|how)\s+(uso|combinar|use)\s+(una?\s+)?(app|ia|chatbot)',
            r'(c[oó]mo|how).{0,30}sin\s+sustituir\s+al?\s+m[eé]dico',
            r'(puede|can)\s+(una?\s+)?(app|ia)\s+orientarme\s+sin\s+diagnosticar',
            r'(c[oó]mo|how).{0,20}engancharme.{0,20}(app|ia)',
            r'(c[oó]mo|how)\s+usar\s+la\s+tecnolog[ií]a\s+para\s+cuidarme',
            r'(c[oó]mo|how)\s+combinar\s+la\s+informaci[oó]n\s+de\s+una?\s+(app|ia)',
            r'l[ií]mites\s+de\s+la\s+ia',
            r'limits?\s+of\s+(ai|the\s+chatbot)',
        ]
        if any(re.search(p, q) for p in general_patterns):
            return 'general'

        return None

    def _setup_templates(self):
        # ─────────────────────────────────────────────────────────────────
        # SHARED SYSTEM PREAMBLE
        # ─────────────────────────────────────────────────────────────────
        # ✨ v3.8: enriched with Alicia's exact domain description +
        # anti-pattern rules for Spanish quality, metaphor handling,
        # AI-meta honesty, and varied closings.
        # IMPORTANT: this preamble is in Spanish (not English) to prevent
        # the LLM from "leaking" English vocabulary into Spanish responses.
        system_preamble = (
            "Eres una asistente especializada en SALUD HORMONAL FEMENINA, "
            "diseñada para acompañar a las mujeres a lo largo de toda su vida hormonal.\n\n"
            "Tus áreas de conocimiento son:\n"
            "- Ciclo menstrual y sus trastornos (síndrome premenstrual, dismenorrea, "
            "amenorrea, sangrados anómalos)\n"
            "- Fertilidad y salud reproductiva\n"
            "- Perimenopausia, menopausia y postmenopausia\n"
            "- Condiciones uterinas y ováricas (endometriosis, miomas, adenomiosis, "
            "síndrome de ovario poliquístico, quistes ováricos)\n"
            "- Salud tiroidea en mujeres y su relación con el ciclo hormonal\n"
            "- Cambios en el metabolismo asociados a las distintas fases del ciclo "
            "hormonal y de la vida (pubertad, edad fértil, perimenopausia, postmenopausia)\n"
            "- Síntomas relacionados con cambios hormonales: sueño, ánimo, ansiedad, "
            "estrés, fatiga, libido, niebla mental, cambios de peso y composición corporal\n\n"
            "REGLAS CRÍTICAS DE PRECISIÓN MÉDICA:\n"
            "1. Si la evidencia científica proporcionada no cubre la pregunta, responde de "
            "forma general y honesta, SIN inventar enfermedades, síndromes o nombres médicos.\n"
            "2. NUNCA traduzcas términos médicos en inglés literalmente. Usa terminología "
            "médica correcta en español: 'miomas' no 'fibroides', 'quistes' no 'cistos', "
            "'sueño' no 'dormir', 'hipotiroidismo' es real, 'hiperplasia tiroidea primaria' NO existe.\n"
            "3. Si no estás segura de un dato, di 'consulta con un especialista' en vez de inventar.\n"
            "4. NO cites estudios ni autores que no aparezcan en el contexto científico proporcionado.\n"
            "5. Escribe en ESPAÑOL CORRECTO: cuida la concordancia de género y número "
            "('los niveles' no 'las niveles', 'el ciclo' no 'la ciclo'), usa tildes "
            "('triglicéridos', 'desorientación', 'ejercicio', 'período'), y evita "
            "anglicismos innecesarios.\n\n"
            "REGLAS DE TONO Y REGISTRO:\n"
            "6. Si la pregunta es METAFÓRICA o EMOCIONAL ('voy a ciegas', 'no soy la misma', "
            "'estoy exagerando', 'me siento perdida'), NO la interpretes como síntoma "
            "clínico literal. Responde primero al sentido humano de la pregunta, valida "
            "la experiencia, y solo después ofrece información clínica si aplica.\n"
            "7. Si la pregunta es de naturaleza META sobre la propia IA ('¿hasta dónde "
            "puede ayudar una IA?', '¿puede una app diagnosticarme?'), responde con "
            "HONESTIDAD sobre lo que el sistema puede ofrecer (orientación, evidencia, "
            "acompañamiento informativo) y lo que NO puede (diagnóstico, sustituir al "
            "médico, conocer tu historial).\n"
            "8. NO termines TODAS las respuestas con 'consulta a tu ginecólogo'. "
            "Recomienda consultar a un profesional SOLO cuando: (a) los síntomas requieren "
            "evaluación clínica real, (b) la pregunta pide orientación sobre cuándo acudir, "
            "o (c) hay señales de alarma. Si la pregunta es informativa, cultural o "
            "reflexiva, puedes cerrar de otras formas (con perspectiva, con validación "
            "emocional, o con una invitación a seguir explorando el tema).\n"
            "9. VARÍA los cierres y el tono. Evita repetir las mismas fórmulas en cada "
            "respuesta. Adapta el cierre al contenido y al tono de la pregunta.\n\n"
            "Tono general: empática, profesional, basada en evidencia, cercana sin ser "
            "infantilizante. Informas y orientas — NUNCA sustituyes el juicio médico. "
            "Cuando un síntoma queda fuera de tu dominio, redirige al médico de cabecera "
            "o al especialista adecuado.\n\n"
            "Responde SOLO en {lang_name}.\n"
        )

        # ── LOW RISK ──────────────────────────────────────────────────────────
        # ✨ v3.8: extended with 5 new examples covering blocks 8/9/10/11
        # of Alicia's dataset (meta-IA, metaphor, cultural, cognitive, libido).
        self.low_risk_template = PromptTemplate(
            input_variables=["question", "context", "lang_name"],
            template=(
                system_preamble +
                "Context (PubMed evidence): {context}\n\n"
                "Examples of correct answers:\n"
                "---\n"
                "Q: ¿Cómo afecta el ciclo menstrual al sueño?\n"
                "A: Las fluctuaciones hormonales a lo largo del ciclo influyen en el descanso. "
                "En la fase lútea, el aumento de progesterona puede elevar la temperatura corporal "
                "y dificultar el sueño profundo. Mantener horarios regulares y un ambiente fresco ayuda. "
                "Si el insomnio aparece de forma cíclica o persiste, vale la pena comentarlo con un ginecólogo.\n"
                "---\n"
                "Q: What can help with hot flashes during menopause?\n"
                "A: Hot flashes during menopause are linked to changes in estrogen levels. "
                "Wearing breathable clothing, avoiding alcohol and spicy foods, and managing stress can reduce them. "
                "Regular exercise also tends to help. If they significantly affect your daily life or sleep, "
                "a gynecologist can discuss options including hormone therapy.\n"
                "---\n"
                "Q: Tengo el ciclo un poco irregular, ¿es normal?\n"
                "A: Es común que los ciclos varíen ligeramente, especialmente en momentos de estrés, "
                "cambios de peso, ejercicio intenso o en perimenopausia. "
                "Variaciones de unos días suelen ser normales. "
                "Si la irregularidad es marcada o persistente más de varios meses, "
                "vale la pena consultar con un ginecólogo para descartar causas hormonales.\n"
                "---\n"
                "Q: ¿La menopausia afecta el metabolismo?\n"
                "A: Sí, la disminución de estrógenos durante la menopausia se asocia con cambios metabólicos: "
                "tendencia a acumular grasa abdominal, menor sensibilidad a la insulina y pérdida de masa muscular. "
                "Mantener actividad física regular, en especial entrenamiento de fuerza, y una alimentación equilibrada "
                "ayuda a contrarrestar estos cambios.\n"
                "---\n"
                "Q: ¿Hasta dónde puede ayudarme una IA en mi salud?\n"
                "A: Una asistente como yo puede orientarte sobre síntomas hormonales, "
                "explicarte qué dice la evidencia científica, ayudarte a organizar dudas "
                "antes de una consulta y acompañarte en el proceso. "
                "Lo que no puedo hacer es diagnosticarte, conocer tu historial completo, "
                "examinarte físicamente ni sustituir el criterio de un profesional. "
                "Lo más útil suele ser combinar ambas cosas: usar la información para llegar "
                "mejor preparada a las visitas médicas y entender lo que te explican.\n"
                "---\n"
                "Q: ¿Por qué siento que voy a ciegas en este proceso?\n"
                "A: Esa sensación es muy común y tiene una raíz real: la información sobre "
                "perimenopausia y menopausia históricamente se ha comunicado mal y poco. "
                "Muchas mujeres llegan a esta etapa sin saber qué esperar, y los síntomas "
                "pueden resultar confusos cuando aparecen sin contexto. "
                "Llevar un registro de síntomas, informarte por fuentes fiables y plantear "
                "preguntas concretas en consulta ayuda a recuperar la sensación de control. "
                "No vas a ciegas porque tú falles — vas a ciegas porque el mapa no te lo dieron.\n"
                "---\n"
                "Q: ¿Por qué parece que las famosas no tienen menopausia o se la han curado?\n"
                "A: La menopausia es un proceso biológico universal: todas las mujeres que "
                "viven lo suficiente la atraviesan. Lo que vemos en figuras públicas suele ser "
                "el resultado de acceso a recursos (terapia hormonal supervisada, tratamientos "
                "estéticos, entrenadores, nutricionistas, equipos de imagen) que no significa "
                "que sus síntomas hayan desaparecido, sino que se gestionan o no se muestran. "
                "Comparar tu experiencia con una imagen pública filtrada es injusto contigo. "
                "Lo que tú vives es lo que viven la mayoría de mujeres reales.\n"
                "---\n"
                "Q: ¿Es normal que se me olviden palabras que tengo en la punta de la lengua?\n"
                "A: Sí, es uno de los síntomas más reportados en perimenopausia y menopausia, "
                "y se conoce como niebla mental. Las fluctuaciones de estrógenos afectan a "
                "áreas del cerebro implicadas en la memoria de trabajo y el lenguaje. "
                "No significa que tu memoria a largo plazo esté dañada, sino que el acceso "
                "rápido a las palabras se ralentiza temporalmente. "
                "Dormir bien, hacer ejercicio aeróbico y mantener actividad cognitiva ayudan. "
                "La mayoría de mujeres recuperan agilidad mental en la postmenopausia estable.\n"
                "---\n"
                "Q: ¿Por qué no tengo ganas de sexo si mi pareja me sigue gustando?\n"
                "A: La libido depende de muchos factores y no solo del afecto hacia tu pareja. "
                "En perimenopausia y menopausia, la bajada de estrógenos y testosterona puede "
                "reducir el deseo, además de aumentar la sequedad vaginal y hacer que la "
                "respuesta sexual sea más lenta. El estrés, el cansancio, la calidad del sueño "
                "y el contexto emocional también influyen. No tener ganas no significa que "
                "haya algo mal con tu relación. Si te preocupa, un ginecólogo puede valorar "
                "el componente hormonal — existen opciones desde lubricantes y cambios de "
                "estilo de vida hasta tratamientos específicos.\n"
                "---\n"
                "Same style — no greeting, no label at end:\n"
                "Q: {question}\n"
                "A:"
            )
        )

        # ── MEDIUM RISK ───────────────────────────────────────────────────────
        self.medium_risk_template = PromptTemplate(
            input_variables=["question", "context", "lang_name"],
            template=(
                system_preamble +
                "Context (PubMed evidence): {context}\n\n"
                "Examples of correct answers:\n"
                "---\n"
                "Q: Llevo 3 meses con la regla muy abundante\n"
                "A: Un sangrado menstrual abundante mantenido durante varios meses puede deberse a miomas, "
                "pólipos, alteraciones hormonales o problemas tiroideos, entre otras causas. "
                "Es importante que un ginecólogo te valore esta o la próxima semana, "
                "y conviene también un análisis para descartar anemia.\n"
                "---\n"
                "Q: I've had pelvic pain for 2 weeks during my cycle\n"
                "A: Pelvic pain that lasts two weeks and follows your cycle may be related to ovulation, "
                "endometriosis, ovarian cysts or other gynecological conditions. "
                "It's worth seeing a gynecologist this week for evaluation, especially if the pain is increasing.\n"
                "---\n"
                "Q: Tengo sofocos muy intensos y no duermo\n"
                "A: Los sofocos intensos que interrumpen el sueño son habituales en perimenopausia y menopausia, "
                "pero tienen impacto real en la calidad de vida y en la salud cardiovascular y ósea a largo plazo. "
                "Conviene consultar con un ginecólogo esta semana para valorar opciones, "
                "que pueden incluir cambios en el estilo de vida o terapia hormonal.\n"
                "---\n"
                "Q: ¿La terapia hormonal aumenta el riesgo de cáncer?\n"
                "A: La respuesta es matizada y depende del tipo de terapia, la duración y el perfil "
                "de cada mujer. La terapia hormonal combinada (estrógeno con progestágeno) se ha "
                "asociado a un ligero aumento del riesgo de cáncer de mama en algunos estudios, "
                "mientras que la terapia solo con estrógenos puede aumentar el riesgo de cáncer de "
                "endometrio si la mujer conserva el útero. Por otro lado, la terapia hormonal está "
                "generalmente contraindicada en supervivientes de cáncer de mama. La decisión debe "
                "tomarse con un ginecólogo valorando beneficios y riesgos individuales. "
                "Estoy aquí si quieres seguir hablando del tema.\n"
                "---\n"
                "Same style — no greeting, no label at end:\n"
                "Q: {question}\n"
                "A:"
            )
        )

        # ── HIGH RISK ─────────────────────────────────────────────────────────
        self.high_risk_template = PromptTemplate(
            input_variables=["question", "context", "lang_name"],
            template=(
                system_preamble +
                "Context (PubMed evidence): {context}\n\n"
                "Do NOT name specific diseases. Say symptoms 'may indicate a condition that needs evaluation'.\n\n"
                "Examples of correct answers:\n"
                "---\n"
                "Q: Tengo sangrado vaginal después de la menopausia\n"
                "A: Cualquier sangrado vaginal después de la menopausia requiere evaluación médica sin demora, "
                "porque puede indicar una condición que necesita ser estudiada. "
                "Es importante acudir al ginecólogo esta misma semana, sin esperar a ver si remite por sí solo.\n"
                "---\n"
                "Q: I have severe pelvic pain and heavy bleeding\n"
                "A: Severe pelvic pain combined with heavy bleeding may indicate a condition that requires "
                "urgent evaluation. Please go to the emergency room or see a gynecologist today, "
                "especially if you feel dizzy, weak, or the bleeding is soaking through pads quickly.\n"
                "---\n"
                "Q: Tengo un bulto en el pecho que ha aparecido hace poco\n"
                "A: La aparición reciente de un bulto en el pecho es algo que siempre debe ser evaluado "
                "por un profesional sin demora. Acude esta semana a tu médico o ginecólogo para una exploración, "
                "ya que la valoración temprana es importante.\n"
                "---\n"
                "Same style — no greeting, no phone numbers, no label at end:\n"
                "Q: {question}\n"
                "A:"
            )
        )

          # ── MENTAL HEALTH (hormone-related mood) ──────────────────────────────
          # ✨ v3.8: 2 new examples for emotional validation without
          # pathologizing ("estoy exagerando", "no soy la misma").
        self.mental_health_template = PromptTemplate(
            input_variables=["question", "context", "lang_name"],
            template=(
                "Compassionate health assistant specialized in female hormonal health. "
                "Reply ONLY in {lang_name}.\n"
                "Match the SEVERITY of the response to the severity of what the person describes. "
                "Be aware that hormonal changes (PMS, PMDD, postpartum, perimenopause, menopause) "
                "can significantly affect mood. Do NOT pathologize emotional reflection — "
                "validate the experience first.\n\n"
                "Context (PubMed evidence, use it if relevant — if not, ignore it and respond from general knowledge): {context}\n\n"




                "Examples of correct answers:\n"
                "---\n"
                "Q: Estoy más irritable los días antes de la regla\n"
                "A: Los cambios de humor en los días previos a la menstruación son frecuentes y forman parte "
                "del síndrome premenstrual. Las fluctuaciones de estrógeno y progesterona influyen en el ánimo. "
                "Llevar un registro del ciclo, mantener actividad física regular y cuidar el sueño suelen ayudar. "
                "Si la irritabilidad es muy intensa o afecta a tu día a día, vale la pena consultar con un ginecólogo, "
                "ya que existen formas más severas (TDPM) que tienen tratamiento.\n"
                "---\n"
                "Q: I feel sad and anxious since menopause started\n"
                "A: Mood changes during perimenopause and menopause are common and have a real biological basis "
                "linked to hormonal shifts, not something you're imagining. Sleep, exercise, and connection with "
                "people you trust help. If the sadness or anxiety is intense or persistent, speaking with your "
                "gynecologist or a mental health professional is a good step — there are effective options.\n"
                "---\n"
                "Q: Después del parto me siento muy triste y no me reconozco\n"
                "A: Lo que describes merece atención. Los cambios hormonales del posparto, junto con la falta de sueño "
                "y los nuevos retos, pueden generar tristeza intensa que va más allá de la 'tristeza normal' del posparto. "
                "Te animo a hablar con tu matrona o médico esta semana — la depresión posparto es frecuente y tiene tratamiento. "
                "No tienes que pasar por esto sola.\n"
                "---\n"
                "Q: Llevo semanas con muchísima ansiedad desde que empecé la perimenopausia\n"
                "A: La ansiedad mantenida durante la perimenopausia es real y tiene base biológica en los cambios hormonales. "
                "Llevar varias semanas así es agotador y merece atención. Te animo a hablar con tu ginecólogo esta semana "
                "para valorar el contexto hormonal, y si lo consideras útil, también con un profesional de salud mental. "
                "No tienes que enfrentarlo sola.\n"
                "---\n"
                "Q: ¿Estoy exagerando o realmente me está pasando algo?\n"
                "A: Si lo que sientes es lo bastante intenso como para hacerte dudar, ya merece atención. "
                "Las mujeres muchas veces minimizamos lo que vivimos porque nos han enseñado a hacerlo, "
                "sobre todo en temas hormonales. Lo que describes es real para ti, y eso es suficiente "
                "razón para explorarlo con calma. Llevar un registro de síntomas durante unas semanas "
                "puede ayudarte a ver patrones, y compartirlo con un profesional facilita que te tomen "
                "en serio. No estás exagerando: estás escuchándote.\n"
                "---\n"
                "Q: ¿Por qué siento que no soy la misma?\n"
                "A: Esa sensación es de las más comunes en perimenopausia y menopausia, y tiene base real: "
                "los cambios hormonales afectan al ánimo, a la energía, a cómo procesas las emociones e "
                "incluso a cómo te percibes. No es que hayas perdido quién eres, es que estás atravesando "
                "una transición biológica y vital importante. Muchas mujeres describen esta etapa como "
                "una recolocación: parte de quien eras sigue ahí, y va apareciendo una versión nueva de ti. "
                "Tener apoyo (cercano, profesional o ambos) hace este tránsito más llevadero.\n"
                "---\n"
                "Same style — empathetic, proportional to severity, no diet advice, no label:\n"
                "Q: {question}\n"
                "A:"
            )
        )

        # ── CLARIFICATION ─────────────────────────────────────────────────────
        self.clarification_template = PromptTemplate(
            input_variables=["question", "lang_name"],
            template=(
                "Health assistant specialized in female hormonal health. "
                "Reply ONLY in {lang_name}.\n\n"
                "Examples:\n"
                "---\n"
                "Q: Me encuentro mal\n"
                "A: ¿Puedes contarme qué síntomas tienes y desde cuándo? "
                "¿Tienen relación con tu ciclo menstrual, con la menopausia o con cambios hormonales recientes?\n"
                "---\n"
                "Q: I don't feel well\n"
                "A: Can you tell me what symptoms you have and how long they have been going on? "
                "Are they related to your menstrual cycle, menopause, or recent hormonal changes?\n"
                "---\n"
                "Same style — max 2 questions, no greeting, no label:\n"
                "Q: {question}\n"
                "A:"
            )
        )

    # ─────────────────────────────────────────────────────────────────────
    # LANGUAGE DETECTION (unchanged from v3.3)
    # ─────────────────────────────────────────────────────────────────────
    def detect_language(self, text: str) -> dict:
        text_lower = text.lower()

        if len(re.findall(r'[\u0600-\u06FF]', text)) > 2:
            return {"name": "Arabic", "code": "ar", "is_greeting": False}

        greeting_patterns = {
            'es': [r'^hola\s*[!.?]?$', r'^buenos\s+días\s*[!.?]?$', r'^buenas\s*[!.?]?$'],
            'en': [r'^h(?:ello|i|ey)\s*[!.?]?$', r'^good\s+morning\s*[!.?]?$'],
            'fr': [r'^(?:bonjour|salut|bonsoir)\s*[!.?]?$'],
            'ar': [r'^(?:مرحبا|السلام|أهلا)\s*[!.?]?$']
        }
        for lang_code, patterns in greeting_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text_lower):
                    lang_names = {'es': 'español', 'en': 'English', 'fr': 'français', 'ar': 'Arabic'}
                    return {"name": lang_names[lang_code], "code": lang_code, "is_greeting": True}

        lang_scores = {"es": 0, "en": 0, "fr": 0}
        score_patterns = {
            "es": [r"\b(el|la|los|las|un|una|tengo|me|mi|dolor|fiebre|desde|hace|qué|cómo|por|llevo|siento|duele|estoy)\b"],
            "en": [r"\b(i|my|have|pain|fever|since|feel|what|how|why|the|is|are|been|hurt|ache|feeling)\b"],
            "fr": [r"\b(je|j'ai|mon|ma|mes|douleur|fièvre|depuis|quoi|comment|pourquoi|suis|fait|mal)\b"]
        }
        for lang, regex_list in score_patterns.items():
            for pattern in regex_list:
                lang_scores[lang] += len(re.findall(pattern, text_lower))

        if re.search(r'\b(tengo|desde hace|me duele|llevo)\b', text_lower):
            lang_scores["es"] += 3
        if re.search(r"\b(i've|i'm|i feel|i have)\b", text_lower):
            lang_scores["en"] += 3
        if re.search(r"\b(j'ai|depuis|je me|je suis)\b", text_lower):
            lang_scores["fr"] += 3
        if re.search(r'\b(quels?|sont|principaux|diabète|type)\b', text_lower):
            lang_scores["fr"] += 5
        detected_lang = max(lang_scores, key=lang_scores.get)
        if all(s == 0 for s in lang_scores.values()):
            detected_lang = "es"

        lang_names = {"es": "español", "en": "English", "fr": "français"}
        return {"name": lang_names[detected_lang], "code": detected_lang, "is_greeting": False}

    # ─────────────────────────────────────────────────────────────────────
    # GREETING — reflects specialization
    # ─────────────────────────────────────────────────────────────────────
    def get_greeting_response(self, lang_code: str) -> str:
        greetings = {
            "es": (
                "¡Hola! 👋 Soy tu asistente especializada en salud hormonal femenina. "
                "Puedo ayudarte con consultas sobre ciclo menstrual, fertilidad, "
                "perimenopausia, menopausia, salud tiroidea y los síntomas hormonales "
                "asociados (sueño, ánimo, fatiga, etc.).\n\n"
                "¿En qué puedo ayudarte?"
            ),
            "en": (
                "Hello! 👋 I'm your assistant specialized in female hormonal health. "
                "I can help with menstrual cycle, fertility, perimenopause, menopause, "
                "thyroid health, and related hormonal symptoms (sleep, mood, fatigue, etc.).\n\n"
                "How can I help you?"
            ),
            "fr": (
                "Bonjour ! 👋 Je suis votre assistante spécialisée en santé hormonale féminine. "
                "Je peux vous aider avec le cycle menstruel, la fertilité, la périménopause, "
                "la ménopause et les symptômes hormonaux associés.\n\n"
                "Comment puis-je vous aider ?"
            ),
            "ar": (
                "مرحبا! 👋 أنا مساعدتك المتخصصة في الصحة الهرمونية للمرأة. "
                "يمكنني مساعدتك في مواضيع الدورة الشهرية، الخصوبة، انقطاع الطمث، "
                "صحة الغدة الدرقية، والأعراض الهرمونية المرتبطة.\n\n"
                "كيف يمكنني مساعدتك؟"
            ),
        }
        return greetings.get(lang_code, greetings["es"])

    # ─────────────────────────────────────────────────────────────────────
    # SCOPE DETECTION — NEW in v3.4
    # ─────────────────────────────────────────────────────────────────────
    def _is_in_scope(self, question: str, lang_code: str) -> bool:
        """
        Determines whether the question is within the scope of female
        hormonal health. Conservative approach: when ambiguous, treat
        as in scope to avoid false negatives that frustrate the user.
        """
        q = question.lower()

        # Get keyword list for the detected language (fallback to es+en)
        keywords = self.IN_SCOPE_KEYWORDS.get(lang_code, [])
        if not keywords:
            keywords = self.IN_SCOPE_KEYWORDS['es'] + self.IN_SCOPE_KEYWORDS['en']

        # Direct keyword match → in scope
        for kw in keywords:
            if kw in q:
                return True

        # Short questions that are likely follow-ups (handled by memory)
        # We don't want to block follow-ups like "y eso?" or "what about that?"
        if len(question.split()) <= 5:
            return True

        # Vague symptom expressions — let clarification handler ask for more info
        for patterns in self.vague_symptoms.values():
            if any(re.search(p, q) for p in patterns):
                return True

        # Profile / memory questions ("how old am I", "what conditions do I have")
        if self.memory.is_profile_question(question):
            return True

        return False

    def _is_clearly_out_of_scope(self, question: str, lang_code: str) -> bool:
        """
        Stricter check: detects questions that are CLEARLY about non-female-hormonal
        topics. Used to redirect with confidence.
        """
        q = question.lower()

        # Topics that are clearly outside scope (general medicine, men's health, etc.)
        out_of_scope_topics = [
            # General respiratory / infectious (handles "qué tomo para la gripe", etc.)
            r'\b(gripe|flu|resfriad|cold(?!\s+sore)|covid|neumon[ií]a|pneumonia|bronqu|tos\s+seca|sore\s+throat|garganta)\b',
            # Cardiovascular (unless menopause-related, which would have hit scope keywords)
            r'\b(infarto|heart attack|stroke|ictus|colesterol(?!\s+y\s+menopaus))',
            # Diabetes (general)
            r'\b(diabetes\s+(tipo\s+1|type\s+1|infantil))',
            # Pediatric (children's health)
            r'\b(mi\s+hijo|my\s+son|mon\s+fils|niño|child(?!birth)|kid|enfant)\b',
            # Men-specific
            r'\b(próstata|prostate|disfunción\s+eréctil|erectile)',
            # Trauma / accidents
            r'\b(fractura|fracture|esguince|sprain|herida|wound|quemadura|burn)',
            # General GI (unless cycle-related)
            r'\b(gastritis|úlcera|ulcer|reflujo|reflux|hemorroides|hemorrhoid)',
            # Skin (general dermatology)
            r'\b(verruga|wart|psoriasis|eczema|eccema|acné(?!\s+(hormonal|menstrual)))',
            # Eyes / ENT
            r'\b(conjuntivitis|otitis|sinusitis|amigdalitis|tonsilit)',
            # Generic flu/cold symptoms WITHOUT female context
            r'\bdolor\s+de\s+garganta\b(?!.*(?:cicl|menstrua|menopaus|hormon))',
            r'\bsore\s+throat\b(?!.*(?:cycle|menstrua|menopaus|hormon))',
        ]

        for pattern in out_of_scope_topics:
            if re.search(pattern, q):
                return True

        return False

    # ─────────────────────────────────────────────────────────────────────
    # VAGUE SYMPTOM / CONTEXT OVERRIDES
    # ─────────────────────────────────────────────────────────────────────
    def _is_vague_symptom(self, question: str, lang_code: str) -> bool:
        if len(question.split()) > 8:
            return False
        q = question.lower()
        return any(re.search(p, q) for p in self.vague_symptoms.get(lang_code, []))

    def _context_override(self, question: str, lang_code: str):
        q = question.lower()

        # Mild sadness — keep at LOW risk, treat as safety-sensitive (empathetic)
        mild_sadness_patterns = [
            r'(estoy|me\s+siento|me\s+encuentro)\s+(un\s+poco|algo|un\s+tanto)\s+(triste|bajo|mal)',
            r'(i\s+feel|i\'m|i\s+am)\s+(a\s+bit|kind\s+of|somewhat|a\s+little)\s+(sad|down|blue|low)',
            r'(je\s+me\s+sens|je\s+suis)\s+(un\s+peu|un\s+petit\s+peu)\s+(triste|déprimé)',
        ]
        if any(re.search(p, q) for p in mild_sadness_patterns):
            return RiskLevel.LOW, True

        return None

    # ─────────────────────────────────────────────────────────────────────
    # FOLLOW-UP DETECTION (unchanged from v3.3)
    # ─────────────────────────────────────────────────────────────────────
    def _is_follow_up(self, question: str) -> bool:
        follow_up_patterns = [
            r'^(y|también|además|pero|entonces)\s',
            r'^(and|also|but|so|what about)\s',
            r'^(et|aussi|mais|alors)\s',
            r'(qué\s+más|what\s+else|quoi\s+d\'autre)',
            r'(en\s+ese\s+caso|in\s+that\s+case)',
            r'(me\s+dijiste|you\s+said|vous\s+avez\s+dit)',
            r'(sobre\s+eso|about\s+that|à\s+ce\s+sujet)',
            r'\b(esto|eso|aquello|this|that|it|cela|ça)\b',
            r'\b(afecta|afectaría|affects|affect)\b',
            r'\b(en\s+ese\s+caso|en\s+mi\s+caso|in\s+my\s+case)\b',
            r'(cómo\s+afecta|how\s+does\s+it|comment\s+cela)',
        ]
        q = question.lower().strip()
        if any(re.search(p, q, re.IGNORECASE) for p in follow_up_patterns):
            return True

        if len(question.split()) <= 5 and "?" in question:
            return True
        return False

    def _enrich_with_memory(self, question: str, session_id: str) -> str:
        """Enriches the question with real conversation context. (v3.3 logic preserved)"""
        if not self.memory.has_context(session_id):
            return question

        q = question.lower()

        third_party_patterns = [
            r'\b(mi\s+beb[eé]|mi\s+hij[oa]|mi\s+madr[e]|mi\s+padr[e]|mi\s+abuel[oa])\b',
            r'\b(my\s+baby|my\s+child|my\s+son|my\s+daughter|my\s+mother|my\s+father)\b',
            r'\b(mon\s+bébé|mon\s+enfant|ma\s+mère|mon\s+père)\b',
        ]
        if any(re.search(p, q) for p in third_party_patterns):
            return question

        emergency_keywords = [
            r'\b(emergencia|urgencia|emergency|urgence)\b',
            r'\b(ataque|infarto|stroke|convulsi[oó]n)\b',
            r'\b(llama|llame|call|appelez)\s+(al\s+)?(112|911|médico|doctor)\b',
        ]
        if any(re.search(p, q) for p in emergency_keywords):
            return question

        profile = self.memory.extract_user_profile(session_id)

        if self._is_follow_up(question):
            context = self.memory.get_context_for_prompt(session_id, max_messages=4)
            if context:
                return (
                    f"Conversación previa:\n{context}\n\n"
                    f"Pregunta de seguimiento: {question}\n"
                    f"Responde a la pregunta de seguimiento. "
                    f"No inventes información que el usuario no haya mencionado."
                )
            return question

        medication_question = any(re.search(p, q) for p in [
            r'\b(medicamento|f[aá]rmaco|pastilla|medicine|drug|médicament)\b',
            r'\b(alergi[ao]|allergic?|allergie)\b',
            r'\b(tomar|take|prendre)\b',
            r'\b(interacci[oó]n|interaction)\b',
        ])

        if medication_question and 'conditions' in profile:
            conditions_str = ', '.join(profile['conditions'])
            name_str = f"El usuario se llama {profile['name']}. " if 'name' in profile else ""
            return (
                f"[{name_str}Condiciones médicas mencionadas por el usuario: {conditions_str}]\n"
                f"{question}"
            )

        personal_question = any(re.search(p, q) for p in [
            r'\b(s[íi]ntoma|symptom|symptôme)\b',
            r'\b(me\s+duele|me\s+siento|i\s+feel|j[e\']?\s+me\s+sens)\b',
            r'\b(llevo|desde\s+hace|for\s+\d+\s+days?|depuis)\b',
        ])

        if personal_question and profile:
            profile_parts = []
            if 'name' in profile:
                profile_parts.append(f"El usuario se llama {profile['name']}")
            if 'age' in profile:
                profile_parts.append(f"tiene {profile['age']} años")
            if profile_parts:
                return f"[{', '.join(profile_parts)}]\n{question}"

        return question

    # ─────────────────────────────────────────────────────────────────────
    # MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────────────────
    def answer(self, question: str, session_id: str = "default") -> str:
        start_time = time.time()

        # 1. Emergency detection (highest priority — never blocked by scope)
        is_emergency, emergency_category = EmergencyHandler.is_emergency(question)
        if is_emergency:
            emergency_response = EmergencyHandler.get_emergency_response(emergency_category, question)
            lang_info = self.detect_language(question)
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", emergency_response)
            self._log_interaction(
                question=question, answer=emergency_response, lang_info=lang_info,
                category=f"EMERGENCY_{emergency_category.upper()}",
                response_time=time.time() - start_time,
                risk_assessment={"risk_level": RiskLevel.HIGH}
            )
            return emergency_response

        # 2. Language detection + greeting
        lang_info = self.detect_language(question)
        if lang_info.get("is_greeting"):
            response = self.get_greeting_response(lang_info["code"])
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="greeting", response_time=time.time() - start_time)
            return response

        # 3. Profile question ("¿qué edad tengo?", "resúmeme mi historial")
        if self.memory.is_profile_question(question):
            response = self.memory.build_profile_response(session_id, lang_info["code"])
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="profile_question", response_time=time.time() - start_time)
            return response

        # 4. ✨ NEW v3.6: AI META-QUESTIONS (Block 8 of Alicia's dataset) ✨
        # Questions about the chatbot itself get curated answers, not LLM-generated ones.
        meta_key = self._is_ai_meta_question(question, lang_info["code"])
        if meta_key:
            lang = lang_info["code"] if lang_info["code"] in self.ai_meta_responses else 'es'
            response = self.ai_meta_responses[lang].get(
                meta_key, self.ai_meta_responses[lang]['general']
            )
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category=f"ai_meta_{meta_key}", response_time=time.time() - start_time)
            return response

        # 5. ✨ NEW v3.4: OUT-OF-SCOPE DETECTION ✨
        # Only redirect when the question is CLEARLY out of scope.
        # Follow-ups, vague symptoms and ambiguous questions go through normal flow.
        if self._is_clearly_out_of_scope(question, lang_info["code"]) \
                and not self._is_follow_up(question):
            response = self.out_of_scope_responses.get(
                lang_info["code"], self.out_of_scope_responses['es']
            )
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="out_of_scope", response_time=time.time() - start_time)
            return response

        # 6. Vague symptom → ask clarification
        if self._is_vague_symptom(question, lang_info["code"]):
            response = self._ask_clarification(question, lang_info)
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="clarification", response_time=time.time() - start_time)
            return response

        # 7. Pre-safety check (crisis, self-harm, dosage requests)
        safety_override = self._pre_safety_check(question, lang_info["code"])
        if safety_override:
            response = self._final_cleanup(safety_override, is_emergency_response=False)
            self.memory.add_message(session_id, "user", question)
            self.memory.add_message(session_id, "assistant", response)
            self._log_interaction(question=question, answer=response, lang_info=lang_info,
                                  category="safety_override", response_time=time.time() - start_time)
            return response

        # 8. Risk assessment
        context_result = self._context_override(question, lang_info["code"])
        if context_result:
            risk_level, is_safety_sensitive = context_result
        else:
            risk_assessment = self.risk_agent.assess(question)
            risk_level = risk_assessment["risk_level"]
            is_safety_sensitive = risk_assessment.get("is_safety_sensitive", False)

        # 9. RAG search (PubMed papers on female hormonal health)
        rag_data = self.rag_agent.search(question, lang_info["code"], top_k=3)
        has_high_confidence = rag_data.get("has_high_confidence", False)
        has_any_results = bool(rag_data['results'])

        # ✨ v3.7: Build context based on confidence level
        if has_any_results:
            # We have papers — use them as evidence
            context = self.rag_agent.format_context(rag_data['results'])
            evidence_quality = "high" if has_high_confidence else "moderate"
        else:
            # No relevant papers found — be HONEST instead of letting LLM invent
            context = (
                "ATENCIÓN: No se han encontrado estudios PubMed específicos sobre esta pregunta "
                "en el knowledge base. Responde de forma GENERAL y HONESTA, sin citar estudios "
                "específicos, sin inventar nombres de enfermedades o síndromes, y recomienda "
                "consultar con un especialista para información personalizada."
            )
            evidence_quality = "none"

        # 10. Memory enrichment + LLM response
        enriched_question = self._enrich_with_memory(question, session_id)
        response = self._generate_response(
            question=enriched_question, risk_level=risk_level,
            is_safety_sensitive=is_safety_sensitive,
            context=context, lang_info=lang_info
        )

        # 11. Post-safety filter + closing + cleanup
        response = self._post_safety_filter(response, lang_info["code"])

        if not is_safety_sensitive:
            response = self._add_closing(response, risk_level, lang_info["code"])

        response = self._final_cleanup(response, is_emergency_response=False)

        # 12. Append PubMed citations (if available)
        # 12. Append PubMed citations (if available) — v3.9: honest fallback when no papers
        if rag_data['results'] and rag_data.get('citations'):
            response += "\n\n" + rag_data['citations']
        else:
            # Honest disclosure: no relevant PubMed papers found for this question
            no_evidence_notes = {
                'es': "\n\n📚 *Para esta pregunta no se han encontrado estudios PubMed directamente relevantes en la base de conocimiento. La respuesta se basa en conocimiento general sobre salud hormonal femenina.*",
                'en': "\n\n📚 *No PubMed studies directly relevant to this question were found in the knowledge base. The answer is based on general knowledge about female hormonal health.*",
                'fr': "\n\n📚 *Aucune étude PubMed directement pertinente n'a été trouvée dans la base de connaissances pour cette question. La réponse repose sur des connaissances générales sur la santé hormonale féminine.*",
                'ar': "\n\n📚 *لم يتم العثور على دراسات PubMed ذات صلة مباشرة بهذا السؤال في قاعدة المعرفة. تستند الإجابة إلى المعرفة العامة حول الصحة الهرمونية للمرأة.*",
            }
            response += no_evidence_notes.get(lang_info["code"], no_evidence_notes['es'])

        # 13. Save to memory + log
        self.memory.add_message(session_id, "user", question)
        self.memory.add_message(session_id, "assistant", response)

        self._log_interaction(
            question=question, answer=response, lang_info=lang_info,
            category=risk_level.value, response_time=time.time() - start_time,
            risk_assessment={"risk_level": risk_level, "is_safety_sensitive": is_safety_sensitive},
            rag_results=rag_data['results'],
            quality_metrics={
                "evidence_quality": evidence_quality,
                "rag_top_score": rag_data.get("top_score", 0),
                "rag_results_count": rag_data["results_count"],
                "has_citations": bool(rag_data.get("citations")),
            }
        )
        return response

    # ─────────────────────────────────────────────────────────────────────
    # RESPONSE GENERATION
    # ─────────────────────────────────────────────────────────────────────
    def _ask_clarification(self, question: str, lang_info: dict) -> str:
        chain = self.clarification_template | self.llm
        r = chain.invoke({"question": question, "lang_name": lang_info["name"]})
        return r.content if hasattr(r, 'content') else str(r)

    def _generate_response(self, question: str, risk_level: RiskLevel,
                           is_safety_sensitive: bool, context: str, lang_info: dict) -> str:
        if is_safety_sensitive:
            chain = self.mental_health_template | self.llm
            r = chain.invoke({"question": question, "context": context, "lang_name": lang_info["name"]})
            return r.content if hasattr(r, 'content') else str(r)
        
        if risk_level == RiskLevel.LOW:
            chain = self.low_risk_template | self.llm
            r = chain.invoke({"question": question, "context": context, "lang_name": lang_info["name"]})
        elif risk_level == RiskLevel.MEDIUM:
            chain = self.medium_risk_template | self.llm
            r = chain.invoke({"question": question, "context": context, "lang_name": lang_info["name"]})
        else:
            chain = self.high_risk_template | self.llm
            r = chain.invoke({"question": question, "context": context, "lang_name": lang_info["name"]})

        return r.content if hasattr(r, 'content') else str(r)

    # ─────────────────────────────────────────────────────────────────────
    # PRE-SAFETY CHECK (unchanged from v3.3)
    # ─────────────────────────────────────────────────────────────────────
    def _pre_safety_check(self, question: str, lang_code: str):
        q = question.lower()

        if any(re.search(p, q) for p in [
            r'(pastillas?|pills?|medicamento|paracetamol|ibuprofeno?).{0,30}(morir|matar|suicid|die|kill|death|muerte)',
            r'(morir|matar|suicid|die|kill|death|muerte).{0,30}(pastillas?|pills?|medicamento|paracetamol|ibuprofeno?)',
            r'(cu[aá]ntas?.{0,10}pastillas?|how\s+many\s+pills?).{0,20}(morir|die|kill|muerte)',
            r'(dose|dosis|cantidad).{0,20}(letal|mortal|lethal|fatal)',
            r'(?:sobredosis|overdose|surdosage)',
        ]):
            return {
                'es': (
                    "💙 Lo que describes me preocupa y quiero que sepas que no estás sola.\n\n"
                    "Por favor, contacta ahora con los servicios de emergencia o una línea de apoyo en crisis. "
                    "Hay personas preparadas para escucharte y ayudarte en este momento."
                ),
                'en': (
                    "💙 What you're describing concerns me deeply, and I want you to know you're not alone.\n\n"
                    "Please reach out to emergency services or a crisis support line right now. "
                    "There are people ready to listen and help you through this."
                ),
                'fr': (
                    "💙 Ce que vous décrivez m'inquiète et je veux que vous sachiez que vous n'êtes pas seule.\n\n"
                    "Veuillez contacter les services d'urgence ou une ligne de crise maintenant. "
                    "Des personnes sont prêtes à vous écouter et à vous aider."
                ),
                'ar': (
                    "💙 ما تصفينه يقلقني وأريدك أن تعرفي أنك لست وحدك.\n\n"
                    "يرجى التواصل مع خدمات الطوارئ أو خط دعم الأزمات الآن."
                )
            }.get(lang_code, "💙 Please contact emergency services or a crisis support line now. You are not alone.")

        if any(re.search(p, q) for p in [
            r'\d+\s*(?:ibuprofeno?|paracetamol|aspirin\w*|pastillas?|pills?)',
            r'(?:puedo|can\s+i|puis-je).{0,20}(?:tomar|take|prendre).{0,20}\d+',
        ]):
            return {
                'es': "No puedo recomendar dosis específicas de medicamentos. Consulta con tu farmacéutico o médico. 💙",
                'en': "I cannot recommend specific medication doses. Please consult your pharmacist or doctor. 💙",
                'fr': "Je ne peux pas recommander des doses spécifiques. Consultez votre pharmacien ou médecin. 💙",
                'ar': "لا يمكنني التوصية بجرعات دواء محددة. يرجى استشارة طبيبك أو صيدلاني. 💙"
            }.get(lang_code, "I cannot recommend specific medication doses. Please consult a healthcare professional. 💙")

        if any(re.search(p, q) for p in [
            r'(suicid|self.?harm|automutil)',
            r'(quiero|want|veux)\s+(morir|die|mourir|matarme|hacerme\s+daño)',
            r'(no\s+quiero\s+vivir|don\'?t\s+want\s+to\s+live)',
            r'(quiero|voy\s+a)\s+hacerme\s+(daño|daño)',
        ]):
            return {
                'es': (
                    "💙 Lo que describes me preocupa y quiero que sepas que no estás sola.\n\n"
                    "Por favor, contacta ahora con los servicios de emergencia o una línea de apoyo en crisis."
                ),
                'en': (
                    "💙 What you're describing concerns me deeply, and I want you to know you're not alone.\n\n"
                    "Please reach out to emergency services or a crisis support line right now."
                ),
                'fr': (
                    "💙 Ce que vous décrivez m'inquiète et je veux que vous sachiez que vous n'êtes pas seule.\n\n"
                    "Veuillez contacter les services d'urgence ou une ligne de crise maintenant."
                ),
                'ar': (
                    "💙 ما تصفينه يقلقني. يرجى التواصل مع خدمات الطوارئ الآن."
                )
            }.get(lang_code, "💙 Please contact emergency services or a crisis support line now. You are not alone.")

        if any(re.search(p, q) for p in [
            r'(dime\s+qué\s+tengo|just\s+tell\s+me\s+what\s+i\s+have)',
            r'(no\s+me\s+mandes\s+al\s+médico|don\'?t\s+send\s+me\s+to\s+the\s+doctor)',
        ]):
            return {
                'es': "No puedo hacer diagnósticos, pero puedo orientarte sobre la urgencia de tus síntomas. ¿Qué síntomas tienes exactamente?",
                'en': "I can't make diagnoses, but I can guide you on urgency. What exactly are your symptoms?",
                'fr': "Je ne peux pas faire de diagnostics, mais je peux vous guider. Quels sont vos symptômes?",
                'ar': "لا يمكنني التشخيص، لكن يمكنني توجيهك. ما هي أعراضك بالضبط؟"
            }.get(lang_code, "I can't diagnose, but I can guide you. What are your symptoms?")

        return None

    # ─────────────────────────────────────────────────────────────────────
    # POST-SAFETY FILTER (unchanged from v3.3)
    # ─────────────────────────────────────────────────────────────────────
    def _post_safety_filter(self, response: str, lang_code: str) -> str:
        for pattern, replacement in [
            (r'\byou have\b(?!\s+(?:the\s+right|a\s+point|enough))', 'you may have'),
            (r'\byou\'re suffering from\b', 'you may be experiencing'),
            (r'\btienes\b(?!\s+(?:razón|que|suerte|tiempo))', 'puedes tener'),
            (r'\bvous avez\b(?!\s+(?:besoin|raison))', 'vous pouvez avoir'),
            (r'\b(diagnosed with|diagnosticado con)\b', 'may indicate'),
        ]:
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)

        response = re.sub(r'Sentence\s+\d+:\s*', '', response, flags=re.IGNORECASE)
        response = re.sub(r'^\s*[-*]\s+', '', response, flags=re.MULTILINE)
        response = re.sub(r'\n{3,}', '\n\n', response)
        response = re.sub(r'\s+', ' ', response).strip()

        return response

    def _add_closing(self, response: str, risk_level: RiskLevel, lang_code: str) -> str:
        if risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM):
            closing = random.choice(self.closings.get(lang_code, self.closings['es']))
            if not response.endswith(closing):
                response += f" {closing}"
        return response

    def _final_cleanup(self, response: str, is_emergency_response: bool = False) -> str:
        for pat in [
            r'^estimad[ao]\s+paciente[,.]?\s*',
            r'^estimada?\s*[,.]?\s*',
            r'^dear\s+patient[,.]?\s*',
            r'^dear[,.]?\s*',
            r'^cher[,.]?\s*',
        ]:
            response = re.sub(pat, '', response, flags=re.IGNORECASE).strip()

        for pat in [
            r'\s*[🔴🟡🟢🟠]\s*(urgente|semi.urgente|no\s+urgente|clasificaci[oó]n)[^.]*\.?$',
            r'\s*(clasificaci[oó]n|classification)\s*:\s*[^\n.]+\.?$',
            r'\s*Urgente\s*[—\-]\s*busca\s+atenci[oó]n[^.]*\.?$',
            r'\s*Semi.urgente\s*[—\-][^.]*\.?$',
            r'\s*No\s+urgente[^.]*\.?$',
            r'\s*🔴[^.]*\.?$',
            r'\s*🟡[^.]*\.?$',
            r'\s*🟢[^.]*\.?$',
        ]:
            response = re.sub(pat, '', response, flags=re.IGNORECASE | re.MULTILINE).strip()

        if not is_emergency_response:
            response = re.sub(r'\b(112|911|15|3114|988|024|717\s*003\s*717)\b', '', response)

        response = re.sub(r'\s{2,}', ' ', response).strip()

        if response and response[-1] not in '.!?':
            response += '.'

        return response

    # ─────────────────────────────────────────────────────────────────────
    # LOGGING (unchanged from v3.3)
    # ─────────────────────────────────────────────────────────────────────
    def _log_interaction(self, question: str, answer: str, lang_info: dict,
                         category: str, response_time: float,
                         risk_assessment: dict = None, rag_results: list = None,
                         quality_metrics: dict = None):
        metadata = {
            "model": "mistral",
            "lang_code": lang_info.get("code", "unknown"),
            "system_version": "v3.9",
            "domain": "female_hormonal_health",
        }
        if risk_assessment:
            metadata["risk_level"] = risk_assessment["risk_level"].value if hasattr(risk_assessment["risk_level"], 'value') else str(risk_assessment["risk_level"])
            metadata["is_safety_sensitive"] = risk_assessment.get("is_safety_sensitive", False)
        if rag_results:
            metadata["rag_used"] = True
        if quality_metrics:
            metadata["quality"] = quality_metrics

        self.logger.log_interaction(
            question=question, answer=answer,
            language=lang_info.get("name", "unknown"),
            category=category, response_time=response_time,
            metadata=metadata
        )


# ─────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────
def health_answer(question: str, session_id: str = "default") -> str:
    return get_agent().answer(question, session_id=session_id)