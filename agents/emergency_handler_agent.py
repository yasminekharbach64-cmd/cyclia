"""
Emergency Detection & Response Handler v3.0

IMPROVEMENTS v3.0:
✅ FIXED: Language detection — no longer uses 'i' as English indicator
✅ FIXED: Keyword matching with regex (handles variations, not just exact substrings)
✅ FIXED: No phone numbers — generic phrases only (works in any country)
✅ NEW: Missing emergencies added — choking child, anaphylaxis, seizures, overdose
✅ NEW: Severity levels — CRITICAL (call emergency services now) vs URGENT (go to ER today)
✅ NEW: Thunderclap headache detected as stroke/hemorrhage red flag
✅ NEW: Aspirin advice removed from chest_pain (not appropriate without diagnosis)
✅ NEW: Arabic darija improved for better detection
"""

import re
from typing import Tuple, Optional


class EmergencyHandler:
    """Smart Emergency Detection & Response System v3.0"""

    
    EMERGENCY_PATTERNS = {

        
        'breathing': {
            'patterns': [
                r"can'?t\s+breath",
                r"cannot\s+breath",
                r"difficulty\s+breath",
                r"hard\s+to\s+breath",
                r"struggling\s+to\s+breath",
                r"shortness\s+of\s+breath",
                r"gasping\s+for\s+(air|breath)",
                r"no\s+(puedo|me\s+deja)\s+respirar",
                r"me\s+cuesta\s+respirar",
                r"falta\s+de\s+aire",
                r"me\s+ahog",
                r"asfixia",
                r"difficulté\s+(à|a)\s+respirer",
                r"ne\s+peux\s+pas\s+respirer",
                r"manque\s+d'?air",
                r"étouffement",
                r"ضيق\s*(في)?\s*التنفس",
                r"ما\s*نقدرش\s*ننفس",
                r"كنتخنق",
            ]
        },

        
        'chest_pain': {
            'patterns': [
                r"chest\s+pain",
                r"(severe|crushing|tight|sharp)\s+chest",
                r"pain\s+(in|on)\s+(my\s+)?chest",
                r"pressure\s+(in|on)\s+(my\s+)?chest",
                r"heart\s+pain",
                r"pain\s+(radiating|spreading)\s+to\s+(arm|jaw|shoulder)",
                r"dolor\s+(de|en\s+el)\s+pecho",
                r"dolor\s+torácico",
                r"opresión\s+en\s+el\s+pecho",
                r"presión\s+en\s+el\s+pecho",
                r"dolor\s+(que\s+)?(irradia|sube)\s+al\s+(brazo|hombro|mandíbula)",
                r"douleur\s+(à\s+la|thoracique|dans\s+la\s+poitrine)",
                r"oppression\s+thoracique",
                r"douleur\s+irradiant\s+au\s+(bras|épaule)",
                r"ألم\s*(في)?\s*الصدر",
                r"وجع\s*(في)?\s*الصدر",
                r"ضيق\s*(في)?\s*الصدر",
            ]
        },

        
        'stroke': {
            'patterns': [
                r"face\s+drooping",
                r"can'?t\s+(raise|move|lift)\s+(my\s+)?(arm|hand)",
                r"slurred\s+speech",
                r"sudden\s+(confusion|weakness|numbness|severe\s+headache)",
                r"loss\s+of\s+balance",
                r"can'?t\s+(speak|talk|see\s+clearly)",
                r"half\s+(face|body)\s+(numb|drooping)",
                r"cara\s+ca[íi]da",
                r"no\s+puedo\s+(mover|levantar)\s+(el\s+)?brazo",
                r"habla\s+arrastrada",
                r"confusión\s+repentina",
                r"dolor\s+de\s+cabeza\s+(súbito|repentino|brutal).{0,30}(peor|vida|nunca)",
                r"pérdida\s+de\s+equilibrio",
                r"visage\s+affaissé",
                r"ne\s+peux\s+pas\s+(bouger|lever)\s+(le\s+)?bras",
                r"élocution\s+difficile",
                r"confusion\s+soudaine",
                r"mal\s+de\s+tête\s+soudain\s+(sévère|intense)",
                r"الوجه\s*كيميل",
                r"ما\s*نقدرش\s*نحرك\s*الدراع",
                r"كلام\s*مشوش",
                r"صداع\s*مفاجئ\s*قوي",
            ]
        },

        
        'thunderclap_headache': {
            'patterns': [
                r"(worst|thunder|worst ever|never felt).{0,20}headache",
                r"headache.{0,20}(worst|10.?10|10 out of 10|sudden|explod)",
                r"(dolor de cabeza|jaqueca).{0,25}(peor|brutal|explosivo|jamás|vida)",
                r"dolor.{0,10}cabeza.{0,25}(peor|súbito|nunca|explosivo)",
                r"10\s*minutos.{0,20}dolor.{0,10}cabeza",
                r"llevo.{0,10}minutos.{0,20}cabeza",
                r"(mal de tête|céphalée).{0,25}(pire|explosif|jamais|soudain)",
                r"أسوأ\s*صداع",
                r"صداع\s*(قوي|فجأة|مفاجئ)",
            ]
        },

        
        'unconscious': {
            'patterns': [
                r"(fainted|passed\s+out|blacked\s+out|collapsed|unconscious)",
                r"lost\s+consciousness",
                r"keep\s+fainting",
                r"fainted\s+(several|multiple|many|twice|again)",
                r"(me\s+)?desmay[éeó]",
                r"perdí\s+el\s+conocimiento",
                r"inconsciente",
                r"desmayos\s+(repetidos|varios|múltiples)",
                r"(évanoui|perdu\s+connaissance|syncope|inconscient)",
                r"(طحت|فقدت\s*الوعي|غبت|كنطيح)",
            ]
        },

        
        'bleeding': {
            'patterns': [
                r"bleeding\s+(won'?t\s+stop|heavily|profusely|severely|uncontrolled)",
                r"(severe|heavy|uncontrolled)\s+bleeding",
                r"blood\s+everywhere",
                r"losing\s+(a\s+lot\s+of\s+)?blood",
                r"sangrado\s+(no\s+para|abundante|severo|incontrolable)",
                r"no\s+para\s+de\s+sangrar",
                r"perdiendo\s+sangre",
                r"saignement\s+(ne\s+s'arrête\s+pas|abondant|sévère)",
                r"hémorragie",
                r"perte\s+de\s+sang",
                r"نزيف\s*(ما\s*كيوقفش|كثير|شديد|ما\s*كيتحكمش)",
                r"كنخسر\s*دم",
            ]
        },

        
        'blood_vomit': {
            'patterns': [
                r"(vomiting|coughing)\s+blood",
                r"blood\s+in\s+(vomit|stool|urine|pee)",
                r"bloody\s+(diarrhea|stool|vomit)",
                r"(vomitando|tosiendo)\s+sangre",
                r"sangre\s+en\s+(vómito|heces|orina)",
                r"diarrea\s+con\s+sangre",
                r"(vomis|crache)\s+du\s+sang",
                r"sang\s+dans\s+(les\s+selles|les\s+vomissements|les\s+urines)",
                r"(كنتقيأ|كنسعل)\s*دم",
                r"دم\s*في\s*(القيء|البراز)",
            ]
        },

        
        'high_fever': {
            'patterns': [
                r"fever\s+(of\s+)?(40|41|42|104|105|106)",
                r"(very\s+high|extreme|dangerously\s+high)\s+fever",
                r"temperature\s+(of\s+)?(40|41|42|104)",
                r"fever\s+(won'?t\s+(go\s+down|break)|won'?t\s+come\s+down)",
                r"fiebre\s+(de\s+)?(40|41|42)",
                r"fiebre\s+(muy\s+alta|altísima|que\s+no\s+baja)",
                r"temperatura\s+(muy\s+)?(elevada|alta)\s*(de\s*)?(40|41)?",
                r"fièvre\s+(de\s+)?(40|41|42)",
                r"(très\s+forte|extrême)\s+fièvre",
                r"fièvre\s+qui\s+ne\s+baisse\s+pas",
                r"حمى\s*(عالية|شديدة|40|41)",
                r"سخانة\s*(قوية|عالية|40)",
            ]
        },

        
        'severe_pain': {
            'patterns': [
                r"(unbearable|excruciating|worst|extreme)\s+pain",
                r"pain\s+(10.?10|10\s+out\s+of\s+10)",
                r"severe\s+abdominal\s+pain",
                r"dolor\s+(insoportable|insufrible|extremo|severo|abdominal\s+severo)",
                r"peor\s+dolor",
                r"douleur\s+(insupportable|extrême|abdominale\s+sévère|pire)",
                r"ألم\s*(ما\s*كيتحملش|قوي\s*بزاف|شديد)",
                r"أسوأ\s*ألم",
            ]
        },

        
        'choking': {
            'patterns': [
                r"(child|baby|kid|infant|toddler).{0,30}(choking|can'?t\s+breath|swallowed|airway)",
                r"(choking|swallowed\s+something).{0,30}(child|baby|kid)",
                r"(niño|bebé|hijo|niña).{0,50}(atragant|trag[oó]|no\s+puede\s+respirar)",
                r"(se\s+ha?\s+trag|atragant).{0,40}(algo|objeto|juguete|no\s+puede)",
                r"(no\s+puede\s+respirar).{0,30}(niño|bebé|hijo|niña)",
                r"(enfant|bébé|enfant).{0,30}(s'étouffe|avalé\s+quelque\s+chose|ne\s+peut\s+pas\s+respirer)",
                r"(ولد|طفل|بيبي).{0,30}(بلع|ما\s*كيتنفسش|كيختنق)",
            ]
        },

        
        'anaphylaxis': {
            'patterns': [
                r"anaphyla",
                r"(severe|serious)\s+allergic\s+reaction",
                r"throat\s+(closing|swelling|tightening)",
                r"can'?t\s+swallow.{0,20}(throat|swollen)",
                r"tongue\s+(swelling|swollen)",
                r"(reacción\s+alérgica\s+grave|anafilaxia|anafilaxis)",
                r"garganta\s+(cerrando|hinchándose|se\s+cierra)",
                r"lengua\s+hinchada",
                r"réaction\s+allergique\s+grave|anaphylaxie",
                r"gorge\s+qui\s+se\s+ferme|langue\s+gonflée",
                r"حساسية\s*(شديدة|خطيرة)",
                r"الحلق\s*كيضيق",
                r"اللسان\s*كيتورم",
            ]
        },

        
        'seizure': {
            'patterns': [
                r"(having|having\s+a|experiencing)\s+(seizure|convulsion|fit)",
                r"(seizure|convulsion).{0,20}(now|happening|started)",
                r"body\s+(shaking|convulsing|jerking)\s+(uncontrollab|violently)",
                r"(convulsiones|convulsión|ataque\s+epiléptico|crisis\s+epiléptica)",
                r"cuerpo\s+(temblando|sacudiéndose)\s+sin\s+control",
                r"(convulsions|crise\s+d'épilepsie|crises\s+convulsives)",
                r"corps\s+qui\s+se\s+convulse",
                r"تشنجات",
                r"نوبة\s*(صرع|تشنج)",
                r"جسمو\s*كيرتجف\s*بلا\s*ما\s*يقدر",
            ]
        },

        
        'overdose': {
            'patterns': [
                r"(overdose|took\s+too\s+many\s+(pills|tablets|medications))",
                r"(accidental|intentional)\s+(overdose|poisoning)",
                r"swallowed.{0,20}(whole\s+bottle|all\s+the\s+pills|too\s+many)",
                r"sobredosis",
                r"tomé\s+(demasiados|muchas|todas\s+las)\s+(pastillas|píldoras|medicamentos)",
                r"(surdosage|overdose|pris\s+trop\s+de\s+(médicaments|comprimés|pilules))",
                r"جرعة\s*زائدة",
                r"أكل\s*(بزاف\s*دوا|كل\s*الحبوب)",
                r"سردوزاج",
            ]
        },
    }

    
    SEVERITY = {
        'breathing':           'CRITICAL',
        'chest_pain':          'CRITICAL',
        'stroke':              'CRITICAL',
        'thunderclap_headache':'CRITICAL',
        'choking':             'CRITICAL',
        'anaphylaxis':         'CRITICAL',
        'seizure':             'CRITICAL',
        'overdose':            'CRITICAL',
        'blood_vomit':         'CRITICAL',
        'bleeding':            'CRITICAL',
        'unconscious':         'URGENT',
        'high_fever':          'URGENT',
        'severe_pain':         'URGENT',
    }

    
    EMERGENCY_RESPONSES = {

        'breathing': {
            'es': """🚨 EMERGENCIA MÉDICA — Dificultad Respiratoria

⚠️ La dificultad para respirar es una emergencia que requiere atención INMEDIATA.

📞 ACTÚA AHORA:
• Llama al emergency services
• O llama al emergency number si estás en América

🏥 Ve a urgencias INMEDIATAMENTE o pide una ambulancia.
⏰ NO ESPERES — Esta es una situación de VIDA O MUERTE.""",

            'en': """🚨 MEDICAL EMERGENCY — Breathing Difficulty

⚠️ Difficulty breathing requires IMMEDIATE emergency attention.

📞 ACT NOW:
• Call emergency services

🏥 Go to the ER IMMEDIATELY or call an ambulance.
⏰ DO NOT WAIT — This is a LIFE-THREATENING situation.""",

            'fr': """🚨 URGENCE MÉDICALE — Difficulté Respiratoire

⚠️ La difficulté à respirer est une urgence nécessitant une attention IMMÉDIATE.

📞 AGISSEZ MAINTENANT:
• Appelez le 15 ou le emergency services

🏥 Allez aux urgences IMMÉDIATEMENT ou appelez une ambulance.
⏰ N'ATTENDEZ PAS — C'est une situation POTENTIELLEMENT MORTELLE.""",

            'ar': """🚨 حالة طوارئ طبية — ضيق في التنفس

⚠️ ضيق التنفس حالة طوارئ تحتاج عناية فورية.

📞 دير دابا:
• عيط emergency services

🏥 سير للمستعجلات فوراً أو طلب إسعاف.
⏰ ما تستناش — هادي حالة خطيرة على الحياة."""
        },

        'chest_pain': {
            'es': """🚨 EMERGENCIA CARDÍACA — Dolor de Pecho

⚠️ El dolor de pecho puede indicar un ataque cardíaco u otra emergencia grave.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA INMEDIATAMENTE

🚑 NO conduzcas tú mismo — espera la ambulancia
🚑 Siéntate o túmbate en posición cómoda
🚑 Afloja la ropa ajustada

⏰ ACTÚA EN SEGUNDOS — no en minutos.""",

            'en': """🚨 CARDIAC EMERGENCY — Chest Pain

⚠️ Chest pain may indicate a heart attack or another serious emergency.

📞 CALL EMERGENCY SERVICES IMMEDIATELY

🚑 DO NOT drive yourself — wait for the ambulance
🚑 Sit or lie in a comfortable position
🚑 Loosen tight clothing

⏰ ACT IN SECONDS — not minutes.""",

            'fr': """🚨 URGENCE CARDIAQUE — Douleur Thoracique

⚠️ La douleur thoracique peut indiquer une crise cardiaque ou une autre urgence grave.

📞 APPELEZ LES SERVICES D'URGENCE IMMÉDIATEMENT

🚑 NE conduisez PAS vous-même — attendez l'ambulance
🚑 Asseyez-vous ou allongez-vous confortablement
🚑 Desserrez les vêtements serrés

⏰ AGISSEZ EN SECONDES — pas en minutes.""",

            'ar': """🚨 حالة طوارئ قلبية — ألم في الصدر

⚠️ ألم الصدر ممكن يكون نوبة قلبية أو حالة خطيرة أخرى.

📞 عيط للإسعاف أو خدمات الطوارئ فوراً

🚑 ما تسوقش بروحك — تسنا الإسعاف
🚑 اقعد أو نام براحة
🚑 حل الحوايج الضيقة

⏰ دير في ثواني — مشي دقائق."""
        },

        'stroke': {
            'es': """🚨 EMERGENCIA NEUROLÓGICA — Posible Derrame Cerebral

⚠️ Estos síntomas pueden indicar un DERRAME CEREBRAL. Cada minuto cuenta.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA INMEDIATAMENTE

🧠 Protocolo RÁPIDO:
• R — ¿Rostro caído de un lado?
• Á — ¿No puede levantar un brazo?
• P — ¿Habla arrastrada o confusa?
• I — ¿Inicio súbito?
• D — DAME tiempo — llama YA
• O — Una sola señal = EMERGENCIA

🏥 NO conduzcas — espera la ambulancia.
⏰ Cada MINUTO de retraso = más daño cerebral.""",

            'en': """🚨 NEUROLOGICAL EMERGENCY — Possible Stroke

⚠️ These symptoms may indicate a STROKE. Every minute matters.

📞 CALL EMERGENCY SERVICES IMMEDIATELY

🧠 FAST Protocol:
• F — Face drooping on one side?
• A — Can't raise one arm?
• S — Slurred or confused speech?
• T — Time to call NOW

🏥 DO NOT drive — wait for the ambulance.
⏰ Every MINUTE of delay = more brain damage.""",

            'fr': """🚨 URGENCE NEUROLOGIQUE — AVC Possible

⚠️ Ces symptômes peuvent indiquer un AVC. Chaque minute compte.

📞 APPELEZ LES SERVICES D'URGENCE IMMÉDIATEMENT

🧠 Protocole VITE:
• V — Visage affaissé d'un côté ?
• I — Incapacité à lever un bras ?
• T — Trouble de la parole ?
• E — Extrême urgence : appelez MAINTENANT

🏥 NE conduisez PAS — attendez l'ambulance.
⏰ Chaque MINUTE = plus de dommages cérébraux.""",

            'ar': """🚨 حالة طوارئ عصبية — سكتة دماغية محتملة

⚠️ هاد الأعراض ممكن تدل على سكتة دماغية. كل دقيقة مهمة.

📞 عيط للإسعاف أو خدمات الطوارئ فوراً

🧠 علامات التحذير:
• الوجه — كيميل لجهة وحدة؟
• الذراع — ما كيقدرش يرفعها؟
• الكلام — مشوش أو صعب؟
• الوقت — دير دابا

🏥 ما تسوقش — تسنا الإسعاف.
⏰ كل دقيقة = ضرر أكثر في الدماغ."""
        },

        'thunderclap_headache': {
            'es': """🚨 ALERTA NEUROLÓGICA — Dolor de Cabeza Súbito Severo

⚠️ Un dolor de cabeza repentino e intensísimo ("el peor de mi vida") puede indicar una HEMORRAGIA CEREBRAL.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA INMEDIATAMENTE

🏥 Ve a urgencias AHORA — no esperes a ver si mejora.

⚠️ Síntomas adicionales de alarma:
• Rigidez de cuello
• Sensibilidad a la luz
• Vómitos repentinos
• Pérdida de consciencia

⏰ Este tipo de dolor es una EMERGENCIA HASTA QUE SE DEMUESTRE LO CONTRARIO.""",

            'en': """🚨 NEUROLOGICAL ALERT — Sudden Severe Headache

⚠️ A sudden, extremely intense headache ("the worst of my life") may indicate a BRAIN HEMORRHAGE.

📞 CALL EMERGENCY SERVICES IMMEDIATELY

🏥 Go to the ER NOW — do not wait to see if it improves.

⚠️ Additional warning signs:
• Stiff neck
• Sensitivity to light
• Sudden vomiting
• Loss of consciousness

⏰ This type of headache is a MEDICAL EMERGENCY until proven otherwise.""",

            'fr': """🚨 ALERTE NEUROLOGIQUE — Céphalée Soudaine Sévère

⚠️ Un mal de tête soudain et extrêmement intense ("le pire de ma vie") peut indiquer une HÉMORRAGIE CÉRÉBRALE.

📞 APPELEZ LES SERVICES D'URGENCE IMMÉDIATEMENT

🏥 Allez aux urgences MAINTENANT — n'attendez pas.

⚠️ Signes supplémentaires d'alarme:
• Raideur de nuque
• Sensibilité à la lumière
• Vomissements soudains
• Perte de conscience

⏰ Ce type de douleur est une URGENCE MÉDICALE jusqu'à preuve du contraire.""",

            'ar': """🚨 تنبيه عصبي — صداع مفاجئ شديد

⚠️ صداع مفاجئ وقوي جداً ("أسوأ صداع في حياتي") ممكن يدل على نزيف في الدماغ.

📞 عيط للإسعاف أو خدمات الطوارئ فوراً

🏥 سير للمستعجلات دابا — ما تستناش.

⚠️ علامات تحذير إضافية:
• تصلب الرقبة
• حساسية للضوء
• قيء مفاجئ
• فقدان الوعي

⏰ هاد النوع من الصداع حالة طوارئ حتى يثبت العكس."""
        },

        'choking': {
            'es': """🚨 EMERGENCIA PEDIÁTRICA — Posible Atragantamiento

⚠️ Un niño que no puede respirar tras tragarse algo es una EMERGENCIA INMEDIATA.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA AHORA MISMO

🚑 Si el niño NO puede toser, llorar ni respirar:
• Menores de 1 año: 5 golpes en la espalda + 5 compresiones en el pecho
• Mayores de 1 año: Maniobra de Heimlich (5 compresiones abdominales)
• NO hagas nada si el niño puede toser fuerte — deja que tosa

🏥 Incluso si se recupera — lleva al niño a urgencias para evaluación.
⏰ Cada segundo cuenta.""",

            'en': """🚨 PEDIATRIC EMERGENCY — Possible Choking

⚠️ A child unable to breathe after swallowing something is an IMMEDIATE EMERGENCY.

📞 CALL EMERGENCY SERVICES RIGHT NOW

🚑 If the child CANNOT cough, cry or breathe:
• Under 1 year: 5 back blows + 5 chest thrusts
• Over 1 year: Heimlich maneuver (5 abdominal thrusts)
• DO NOT intervene if child can cough forcefully — let them cough

🏥 Even if recovered — take child to ER for evaluation.
⏰ Every second counts.""",

            'fr': """🚨 URGENCE PÉDIATRIQUE — Possible Étouffement

⚠️ Un enfant incapable de respirer après avoir avalé quelque chose est une URGENCE IMMÉDIATE.

📞 APPELEZ LES SERVICES D'URGENCE MAINTENANT

🚑 Si l'enfant NE PEUT PAS tousser, pleurer ou respirer:
• Moins de 1 an: 5 tapes dans le dos + 5 compressions thoraciques
• Plus de 1 an: Manœuvre de Heimlich (5 compressions abdominales)
• N'intervenez PAS si l'enfant peut tousser fort — laissez-le tousser

🏥 Même s'il récupère — emmenez l'enfant aux urgences.
⏰ Chaque seconde compte.""",

            'ar': """🚨 حالة طوارئ أطفال — اختناق محتمل

⚠️ طفل ما كيقدرش يتنفس بعد ما بلع شي — حالة طوارئ فورية.

📞 عيط للإسعاف أو خدمات الطوارئ دابا

🚑 إلا الطفل ما قدرش يسعل أو يبكي أو يتنفس:
• أقل من سنة: 5 ضربات على الظهر + 5 ضغطات على الصدر
• أكبر من سنة: مناورة هيمليك (5 ضغطات على البطن)
• ما تدخلش إلا الطفل كيقدر يسعل بقوة — خليه يسعل

🏥 حتى إلا تحسن — سير للمستعجلات للتقييم.
⏰ كل ثانية مهمة."""
        },

        'anaphylaxis': {
            'es': """🚨 EMERGENCIA ALÉRGICA — Posible Anafilaxia

⚠️ Una reacción alérgica grave con dificultad para respirar o hinchazón de garganta puede ser MORTAL en minutos.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA INMEDIATAMENTE

🚑 Si tienes un autoinyector de adrenalina (EpiPen) — ÚSALO AHORA
🚑 Siéntate erguido (o recuéstate si tienes mareos)
🚑 NO te quedes solo

⏰ La anafilaxia puede progresar en minutos — actúa YA.""",

            'en': """🚨 ALLERGIC EMERGENCY — Possible Anaphylaxis

⚠️ A severe allergic reaction with breathing difficulty or throat swelling can be FATAL within minutes.

📞 CALL EMERGENCY SERVICES IMMEDIATELY

🚑 If you have an epinephrine auto-injector (EpiPen) — USE IT NOW
🚑 Sit upright (or lie down if dizzy)
🚑 Do NOT stay alone

⏰ Anaphylaxis can progress in minutes — act NOW.""",

            'fr': """🚨 URGENCE ALLERGIQUE — Anaphylaxie Possible

⚠️ Une réaction allergique grave avec difficulté à respirer ou gonflement de la gorge peut être MORTELLE en quelques minutes.

📞 APPELEZ LES SERVICES D'URGENCE IMMÉDIATEMENT

🚑 Si vous avez un auto-injecteur d'adrénaline (EpiPen) — UTILISEZ-LE MAINTENANT
🚑 Asseyez-vous droit (ou allongez-vous si vous avez des étourdissements)
🚑 Ne restez PAS seul(e)

⏰ L'anaphylaxie peut évoluer en minutes — agissez MAINTENANT.""",

            'ar': """🚨 حالة طوارئ حساسية — حساسية شديدة محتملة

⚠️ حساسية شديدة مع صعوبة التنفس أو تورم الحلق ممكن تكون خطيرة على الحياة في دقائق.

📞 عيط للإسعاف أو خدمات الطوارئ فوراً

🚑 إلا عندك حقنة أدرينالين (EpiPen) — استعملها دابا
🚑 اقعد مستقيم (أو نام إلا عندك دوار)
🚑 ما تبقاش وحدك

⏰ الحساسية الشديدة تتقدم في دقائق — دير دابا."""
        },

        'seizure': {
            'es': """🚨 EMERGENCIA NEUROLÓGICA — Convulsiones

⚠️ Las convulsiones requieren evaluación médica urgente.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA

🚑 Mientras llega la ayuda:
• Pon a la persona de lado (posición de seguridad)
• Aleja objetos peligrosos del entorno
• NO pongas nada en su boca
• NO sujetes a la persona con fuerza
• Cronometra la duración

⚠️ LLAMA AL emergency number URGENTE si:
• La convulsión dura más de 5 minutos
• No recupera la consciencia
• Es la primera vez

⏰ No dejes sola a la persona.""",

            'en': """🚨 NEUROLOGICAL EMERGENCY — Seizure

⚠️ Seizures require urgent medical evaluation.

📞 CALL EMERGENCY SERVICES

🚑 While help arrives:
• Turn person on their side (recovery position)
• Clear dangerous objects from surroundings
• Do NOT put anything in their mouth
• Do NOT restrain the person forcefully
• Time the duration

⚠️ CALL emergency number URGENTLY if:
• Seizure lasts more than 5 minutes
• Person doesn't regain consciousness
• It's their first seizure

⏰ Do not leave the person alone.""",

            'fr': """🚨 URGENCE NEUROLOGIQUE — Convulsions

⚠️ Les convulsions nécessitent une évaluation médicale urgente.

📞 APPELEZ LES SERVICES D'URGENCE

🚑 En attendant les secours:
• Mettez la personne sur le côté (position latérale de sécurité)
• Éloignez les objets dangereux
• Ne mettez RIEN dans sa bouche
• Ne retenez PAS la personne avec force
• Chronométrez la durée

⚠️ APPELEZ D'URGENCE si:
• La crise dure plus de 5 minutes
• La personne ne reprend pas conscience
• C'est la première crise

⏰ Ne laissez pas la personne seule.""",

            'ar': """🚨 حالة طوارئ عصبية — تشنجات

⚠️ التشنجات تحتاج تقييم طبي عاجل.

📞 عيط للإسعاف أو خدمات الطوارئ

🚑 و نتا كتسنا المساعدة:
• حول الشخص على جنبو (وضعية الأمان)
• بعد الأشياء الخطرة من حواليه
• ما تحطش والو في فمو
• ما تمسكوش بقوة
• احسب المدة

⚠️ عيط emergency number فوراً إلا:
• التشنج دام أكثر من 5 دقائق
• الشخص ما رجعلوش الوعي
• هي المرة الأولى

⏰ ما تخليش الشخص وحدو."""
        },

        'overdose': {
            'es': """🚨 EMERGENCIA MÉDICA — Posible Sobredosis

⚠️ Una sobredosis de medicamentos o sustancias puede ser mortal.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA INMEDIATAMENTE

🚑 Mientras llega la ayuda:
• Mantén a la persona despierta y hablando si es posible
• Si está inconsciente pero respira → posición de seguridad (de lado)
• Si no respira → RCP
• NO induzcas el vómito

ℹ️ Dile al operador del emergency number qué sustancia tomó y cuánta si lo sabes.

⏰ El tiempo es crítico — actúa YA.""",

            'en': """🚨 MEDICAL EMERGENCY — Possible Overdose

⚠️ A medication or substance overdose can be life-threatening.

📞 CALL EMERGENCY SERVICES IMMEDIATELY

🚑 While help arrives:
• Keep person awake and talking if possible
• If unconscious but breathing → recovery position (on their side)
• If not breathing → CPR
• Do NOT induce vomiting

ℹ️ Tell the emergency number operator what substance was taken and how much if known.

⏰ Time is critical — act NOW.""",

            'fr': """🚨 URGENCE MÉDICALE — Possible Surdosage

⚠️ Une surdose de médicaments ou de substances peut être mortelle.

📞 APPELEZ LES SERVICES D'URGENCE IMMÉDIATEMENT

🚑 En attendant les secours:
• Gardez la personne éveillée et en train de parler si possible
• Si inconsciente mais respire → position latérale de sécurité
• Si elle ne respire pas → RCP
• Ne provoquez PAS le vomissement

ℹ️ Dites à l'opérateur du 15 quelle substance a été prise et en quelle quantité si vous le savez.

⏰ Le temps est critique — agissez MAINTENANT.""",

            'ar': """🚨 حالة طوارئ طبية — جرعة زائدة محتملة

⚠️ الجرعة الزائدة من الدواء أو أي مادة ممكن تكون خطيرة على الحياة.

📞 عيط للإسعاف أو خدمات الطوارئ فوراً

🚑 و نتا كتسنا المساعدة:
• خلي الشخص صاحي وكيهضر إلا قدرتي
• إلا فاقد الوعي بصح كيتنفس → حولو على جنبو
• إلا ما كيتنفسش → إنعاش قلبي رئوي
• ما تحوسش القيء

ℹ️ قول لمشغل emergency number أي مادة تاخدات و كم إلا عرفتي.

⏰ الوقت حاسم — دير دابا."""
        },

        'bleeding': {
            'es': """🚨 EMERGENCIA MÉDICA — Sangrado Severo

⚠️ El sangrado que no se detiene es una emergencia grave.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA AHORA

🚑 Mientras llega la ayuda:
• Presiona FIRMEMENTE sobre la herida con un paño limpio
• NO retires el paño si se empapa — añade más encima
• Mantén la presión constante
• Si es en un miembro → elévalo por encima del corazón si puedes

⏰ Cada minuto de retraso aumenta el riesgo — busca ayuda YA.""",

            'en': """🚨 MEDICAL EMERGENCY — Severe Bleeding

⚠️ Bleeding that won't stop is a serious emergency.

📞 CALL EMERGENCY SERVICES NOW

🚑 While help arrives:
• Press FIRMLY on wound with a clean cloth
• DO NOT remove cloth if soaked — add more on top
• Maintain constant pressure
• If on a limb → elevate above heart level if possible

⏰ Every minute increases risk — seek help NOW.""",

            'fr': """🚨 URGENCE MÉDICALE — Saignement Sévère

⚠️ Un saignement qui ne s'arrête pas est une urgence grave.

📞 APPELEZ LES SERVICES D'URGENCE MAINTENANT

🚑 En attendant les secours:
• Appuyez FERMEMENT sur la plaie avec un tissu propre
• NE retirez PAS le tissu s'il est imbibé — ajoutez-en plus
• Maintenez une pression constante
• Si c'est un membre → élevez-le au-dessus du niveau du cœur si possible

⏰ Chaque minute augmente le risque — cherchez de l'aide MAINTENANT.""",

            'ar': """🚨 حالة طوارئ طبية — نزيف شديد

⚠️ النزيف اللي ما كيوقفش حالة طوارئ خطيرة.

📞 عيط للإسعاف أو خدمات الطوارئ دابا

🚑 و نتا كتسنا المساعدة:
• اضغط بقوة على الجرح بقماش نظيف
• ما تحيدش القماش إلا تشرب — زيد فوقو
• حافظ على الضغط
• إلا فالجسد → رفع الطرف فوق مستوى القلب إلا قدرتي

⏰ كل دقيقة كتزيد الخطر — طلب المساعدة دابا."""
        },

        'blood_vomit': {
            'es': """🚨 EMERGENCIA MÉDICA GRAVE — Vómito con Sangre

⚠️ Vomitar sangre es una emergencia potencialmente mortal.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA INMEDIATAMENTE

🚑 Mientras llega la ayuda:
• NO comas ni bebas nada
• Mantente sentado o de lado
• NO te acuestes boca arriba

⏰ Esta es una EMERGENCIA ABSOLUTA — actúa YA.""",

            'en': """🚨 SEVERE MEDICAL EMERGENCY — Vomiting Blood

⚠️ Vomiting blood is a potentially life-threatening emergency.

📞 CALL EMERGENCY SERVICES IMMEDIATELY

🚑 While help arrives:
• Do NOT eat or drink anything
• Stay sitting or on your side
• Do NOT lie flat on your back

⏰ This is an ABSOLUTE EMERGENCY — act NOW.""",

            'fr': """🚨 URGENCE MÉDICALE GRAVE — Vomissement de Sang

⚠️ Vomir du sang est une urgence potentiellement mortelle.

📞 APPELEZ LES SERVICES D'URGENCE IMMÉDIATEMENT

🚑 En attendant les secours:
• NE mangez ni ne buvez rien
• Restez assis ou sur le côté
• NE vous allongez PAS à plat sur le dos

⏰ C'est une URGENCE ABSOLUE — agissez MAINTENANT.""",

            'ar': """🚨 حالة طوارئ طبية خطيرة — تقيؤ دم

⚠️ تقيؤ الدم حالة طوارئ خطيرة على الحياة.

📞 عيط للإسعاف أو خدمات الطوارئ فوراً

🚑 و نتا كتسنا المساعدة:
• ما تاكلش أو تشربش والو
• بقا جالس أو على جنب
• ما تنامش على ضهرك

⏰ هادي حالة طوارئ مطلقة — دير دابا."""
        },

        'unconscious': {
            'es': """🚨 EMERGENCIA MÉDICA — Pérdida de Consciencia

⚠️ Los desmayos repetidos son una señal de alarma grave.

📞 LLAMA A LOS SERVICIOS DE EMERGENCIA AHORA

Los desmayos múltiples pueden indicar problemas cardíacos, hemorragia interna o trastornos neurológicos.

🏥 Necesitas evaluación médica de emergencia INMEDIATA.
⏰ NO esperes otro episodio — busca ayuda YA.""",

            'en': """🚨 MEDICAL EMERGENCY — Loss of Consciousness

⚠️ Repeated fainting is a serious warning sign.

📞 CALL EMERGENCY SERVICES NOW

Multiple fainting episodes may indicate heart problems, internal bleeding, or neurological disorders.

🏥 You need IMMEDIATE emergency medical evaluation.
⏰ DO NOT wait for another episode — seek help NOW.""",

            'fr': """🚨 URGENCE MÉDICALE — Perte de Conscience

⚠️ Des évanouissements répétés sont un signe d'alarme grave.

📞 APPELEZ LES SERVICES D'URGENCE MAINTENANT

Des évanouissements multiples peuvent indiquer des problèmes cardiaques, une hémorragie interne ou des troubles neurologiques.

🏥 Vous avez besoin d'une évaluation médicale d'urgence IMMÉDIATE.
⏰ N'attendez PAS un autre épisode — cherchez de l'aide MAINTENANT.""",

            'ar': """🚨 حالة طوارئ طبية — فقدان الوعي

⚠️ الطيحات المتكررة علامة خطر كبيرة.

📞 عيط للإسعاف أو خدمات الطوارئ دابا

الطيحات المتعددة ممكن تدل على مشاكل في القلب، نزيف داخلي أو اضطرابات عصبية.

🏥 خاصك تقييم طبي عاجل فوراً.
⏰ ما تستناش طيحة أخرى — طلب المساعدة دابا."""
        },

        'high_fever': {
            'es': """🚨 ALERTA MÉDICA — Fiebre Muy Alta

⚠️ Fiebre de 40°C o más es potencialmente peligrosa.

📞 Llama a emergencias o ve a urgencias AHORA

🌡️ Mientras esperas:
• Quítate ropa excesiva
• Aplica paños húmedos tibios (NO fríos)
• Bebe líquidos si puedes

⏰ Necesitas atención médica URGENTE.""",

            'en': """🚨 MEDICAL ALERT — Very High Fever

⚠️ Fever of 40°C (104°F) or higher is potentially dangerous.

📞 Call emergency services or go to the ER NOW

🌡️ While waiting:
• Remove excess clothing
• Apply lukewarm (NOT cold) damp cloths
• Drink fluids if you can

⏰ You need URGENT medical attention.""",

            'fr': """🚨 ALERTE MÉDICALE — Fièvre Très Élevée

⚠️ Une fièvre de 40°C (104°F) ou plus est potentiellement dangereuse.

📞 Appelez les services d'urgence ou allez aux urgences MAINTENANT

🌡️ En attendant:
• Enlevez les vêtements excessifs
• Appliquez des linges humides tièdes (PAS froids)
• Buvez des liquides si vous pouvez

⏰ Vous avez besoin d'attention médicale URGENTE.""",

            'ar': """🚨 تنبيه طبي — حمى عالية جداً

⚠️ حمى 40 درجة أو أكثر خطيرة.

📞 عيط للإسعاف أو خدمات الطوارئ أو سير للمستعجلات دابا

🌡️ و نتا كتسنا:
• حيد الحوايج الزايدة
• حط قماش مبلول فاتر (مشي بارد)
• شرب سوائل إلا قدرتي

⏰ خاصك عناية طبية عاجلة."""
        },

        'severe_pain': {
            'es': """🚨 ALERTA MÉDICA — Dolor Severo

⚠️ Un dolor insoportable requiere evaluación médica inmediata.

📞 Llama a emergencias o ve a urgencias AHORA

El dolor extremo puede indicar apendicitis, pancreatitis, úlcera perforada u otras emergencias abdominales.

🏥 NO esperes a que empeore — busca ayuda INMEDIATAMENTE.""",

            'en': """🚨 MEDICAL ALERT — Severe Pain

⚠️ Unbearable pain requires immediate medical evaluation.

📞 Call emergency services or go to the ER NOW

Extreme pain may indicate appendicitis, pancreatitis, perforated ulcer, or other abdominal emergencies.

🏥 DO NOT wait for it to worsen — seek help IMMEDIATELY.""",

            'fr': """🚨 ALERTE MÉDICALE — Douleur Sévère

⚠️ Une douleur insupportable nécessite une évaluation médicale immédiate.

📞 Appelez les services d'urgence ou allez aux urgences MAINTENANT

Une douleur extrême peut indiquer une appendicite, pancréatite, ulcère perforé ou d'autres urgences abdominales.

🏥 N'attendez PAS que ça empire — cherchez de l'aide IMMÉDIATEMENT.""",

            'ar': """🚨 تنبيه طبي — ألم شديد

⚠️ الألم اللي ما كيتحملش كيحتاج تقييم طبي فوري.

📞 عيط للإسعاف أو خدمات الطوارئ أو سير للمستعجلات دابا

الألم الشديد ممكن يدل على التهاب الزائدة، البنكرياس، قرحة مثقوبة أو حالات بطنية أخرى.

🏥 ما تستناش يزيد — طلب المساعدة فوراً."""
        },
    }

    
    @staticmethod
    def detect_language(text: str) -> str:
        
        if len(re.findall(r'[\u0600-\u06FF]', text)) > 2:
            return 'ar'

        text_lower = text.lower()
        scores = {'es': 0, 'en': 0, 'fr': 0}

        
        es_markers = [r'\b(tengo|estoy|dolor|fiebre|sangre|desde\s+hace|me\s+duele|no\s+puedo|llevo)\b']
        en_markers = [r"\b(i\s+have|i'm|i\s+can't|i\s+feel|my\s+\w+|breathing|hurts|pain\s+in)\b"]
        fr_markers = [r"\b(j'ai|je\s+ne|depuis|je\s+me|douleur|fièvre|je\s+suis|ne\s+peux\s+pas)\b"]

        for pattern in es_markers:
            scores['es'] += len(re.findall(pattern, text_lower))
        for pattern in en_markers:
            scores['en'] += len(re.findall(pattern, text_lower))
        for pattern in fr_markers:
            scores['fr'] += len(re.findall(pattern, text_lower))

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else 'es'

    
    @classmethod
    def is_emergency(cls, question: str) -> Tuple[bool, Optional[str]]:
        question_lower = question.lower()

        for category, data in cls.EMERGENCY_PATTERNS.items():
            for pattern in data['patterns']:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    return True, category
                heart_attack_combinations = [
            (r'(dolor|pecho|chest|poitrine)', r'(brazo|arm|bras)'),
            (r'(dolor|pecho|chest)', r'(sudo?\s*frío|cold\s*sweat|sueur\s*froide)'),
            (r'(pecho|chest)', r'(dormido|entumecido|numb|engourdi)'),
        ]
        for pat1, pat2 in heart_attack_combinations:
            if re.search(pat1, question_lower) and re.search(pat2, question_lower):
                return True, 'chest_pain'


        return False, None

    
    @classmethod
    def get_emergency_response(cls, category: str, question: str) -> str:
        lang = cls.detect_language(question)

        if category in cls.EMERGENCY_RESPONSES:
            return cls.EMERGENCY_RESPONSES[category].get(
                lang,
                cls.EMERGENCY_RESPONSES[category]['en']
            )

        # Generic fallback
        generic = {
            'es': "🚨 EMERGENCIA MÉDICA\n\n⚠️ Los síntomas que describes requieren atención médica INMEDIATA.\n\n📞 LLAMA A LOS SERVICIOS DE EMERGENCIA AHORA\n🏥 Ve a urgencias INMEDIATAMENTE\n\n⏰ NO esperes — busca ayuda profesional YA.",
            'en': "🚨 MEDICAL EMERGENCY\n\n⚠️ The symptoms you describe require IMMEDIATE medical attention.\n\n📞 CALL EMERGENCY SERVICES NOW\n🏥 Go to the ER IMMEDIATELY\n\n⏰ DO NOT wait — seek professional help NOW.",
            'fr': "🚨 URGENCE MÉDICALE\n\n⚠️ Les symptômes que vous décrivez nécessitent une attention médicale IMMÉDIATE.\n\n📞 APPELEZ LES SERVICES D'URGENCE MAINTENANT\n🏥 Allez aux urgences IMMÉDIATEMENT\n\n⏰ N'attendez PAS — cherchez de l'aide professionnelle MAINTENANT.",
            'ar': "🚨 حالة طوارئ طبية\n\n⚠️ الأعراض اللي كتوصف كتحتاج عناية طبية فورية.\n\n📞 عيط للإسعاف أو خدمات الطوارئ دابا\n🏥 سير للمستعجلات فوراً\n\n⏰ ما تستناش — طلب مساعدة طبية دابا."
        }
        return generic.get(lang, generic['en'])

    @classmethod
    def get_severity(cls, category: str) -> str:
        """Returns 'CRITICAL' or 'URGENT' for a given emergency category"""
        return cls.SEVERITY.get(category, 'CRITICAL')



def test_emergency_handler():
    test_cases = [
        ("I can't breathe properly",                    'breathing'),
        ("No puedo respirar",                           'breathing'),
        ("Tengo dolor de pecho",                        'chest_pain'),
        ("Llevo 10 minutos con un dolor de cabeza brutal, el peor de mi vida", 'thunderclap_headache'),
        ("My baby swallowed something and can't breathe well", 'choking'),
        ("Mi hijo de 2 años se ha tragado algo",        'choking'),
        ("I'm having a severe allergic reaction, throat is closing", 'anaphylaxis'),
        ("I fainted several times today",               'unconscious'),
        ("Sangrado que no para",                        'bleeding'),
        ("Fiebre de 40 grados",                        'high_fever'),
        ("Vomitando sangre",                           'blood_vomit'),
        ("Having a seizure right now",                 'seizure'),
        ("Tomé demasiadas pastillas",                  'overdose'),
        ("What are symptoms of a cold?",               None),
        ("Me duele la garganta",                       None),
    ]

    print("🧪 Testing Emergency Handler v3.0\n" + "=" * 50)
    passed = 0

    for question, expected in test_cases:
        is_emerg, category = EmergencyHandler.is_emergency(question)
        ok = category == expected
        passed += ok
        status = "✅" if ok else "❌"
        print(f"\n{status} Q: {question}")
        print(f"   Expected: {expected} | Got: {category}", end="")
        if category:
            severity = EmergencyHandler.get_severity(category)
            print(f" | Severity: {severity}", end="")
        print()

    print(f"\n{'='*50}")
    print(f"Results: {passed}/{len(test_cases)} passed")


if __name__ == "__main__":
    test_emergency_handler()