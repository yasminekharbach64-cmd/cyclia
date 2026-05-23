"""
Conversation Memory System v2.1
FIXES:
✅ FIX 1: extract_user_profile() — extrae nombre, edad, condiciones del historial
✅ FIX 2: get_context_for_prompt() — filtra mensajes de terceros (bebé, hijo...)
✅ FIX 3: is_profile_question() — detecta "resúmeme mi perfil", "recuerdas mi nombre"
✅ FIX 4: Perfil real basado SOLO en lo que el usuario dijo (no inventa datos)
✅ FIX 5: name_blacklist — evita capturar "Alérgica", "Diabético" como nombres
✅ FIX 6: get_context_for_prompt() — excluye mensajes sobre terceros del contexto
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path


THIRD_PARTY_PATTERNS = [
    r'\b(mi\s+beb[eé]|mi\s+hij[oa]|mi\s+madr[e]|mi\s+padr[e]|mi\s+abuel[oa]|mi\s+hermano|mi\s+hermana)\b',
    r'\b(my\s+baby|my\s+child|my\s+son|my\s+daughter|my\s+mother|my\s+father|my\s+brother|my\s+sister)\b',
    r'\b(mon\s+bébé|mon\s+enfant|ma\s+mère|mon\s+père|ma\s+sœur|mon\s+frère)\b',
]


def _is_third_party_message(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in THIRD_PARTY_PATTERNS)


class ConversationMemory:
    """Manages conversation history and context"""

    def __init__(self, max_history: int = 10, storage_dir: str = "conversations"):
        self.max_history = max_history
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.active_conversations: Dict[str, List[Dict]] = {}

    def add_message(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        if session_id not in self.active_conversations:
            self.active_conversations[session_id] = []

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        self.active_conversations[session_id].append(message)

        if len(self.active_conversations[session_id]) > self.max_history:
            self.active_conversations[session_id] = self.active_conversations[session_id][-self.max_history:]

        self._save_conversation(session_id)

    def get_history(self, session_id: str, last_n: Optional[int] = None) -> List[Dict]:
        if session_id not in self.active_conversations:
            self._load_conversation(session_id)

        history = self.active_conversations.get(session_id, [])

        if last_n:
            return history[-last_n:]
        return history

    def has_context(self, session_id: str) -> bool:
        history = self.get_history(session_id)
        return len(history) > 0

    def clear_conversation(self, session_id: str):
        if session_id in self.active_conversations:
            del self.active_conversations[session_id]

        conv_file = self.storage_dir / f"{session_id}.json"
        if conv_file.exists():
            conv_file.unlink()

    

    def extract_user_profile(self, session_id: str) -> Dict:
        """
        Extracts real user info from conversation history.
        Returns only what was explicitly mentioned by the user.
        Skips third-party messages (mi bebé, mi hijo, etc.)
        """
        history = self.get_history(session_id)
        user_messages = [
            m["content"] for m in history
            if m["role"] == "user" and not _is_third_party_message(m["content"])
        ]

        profile = {}

        name_blacklist = {
            'alérgica', 'alergica', 'alérgico', 'alergico', 'diabético', 'diabetico',
            'diabética', 'diabetica', 'hipertenso', 'hipertensa', 'asmático', 'asmatico',
            'embarazada', 'enfermo', 'enferma', 'médico', 'medico', 'doctor',
            'bien', 'mal', 'aquí', 'aqui', 'mayor', 'menor', 'es', 'un', 'una',
            'allergic', 'diabetic', 'asthmatic', 'patient', 'nurse',
        }

        name_patterns = [
            r'me\s+llamo\s+([A-Za-záéíóúñÁÉÍÓÚÑ]{3,20})',
            r'mi\s+nombre\s+es\s+([A-Za-záéíóúñÁÉÍÓÚÑ]{3,20})',
            r'my\s+name\s+is\s+([A-Za-z]{3,20})',
            r'je\s+m\'?appelle\s+([A-Za-záéíóúÁÉÍÓÚ]{3,20})',
        ]

        age_patterns = [
            r'tengo\s+(\d+)\s+años',
            r'(\d+)\s+años',
            r"i'?m\s+(\d+)\s+years?\s+old",
            r'(\d+)\s+years?\s+old',
            r"j'?ai\s+(\d+)\s+ans",
        ]

        condition_patterns = [
            (r'\b(diabet[eé]s?|diabético|diabética|diabetic)\b', 'diabetes'),
            (r'\b(hipertens[ií]on|hipertenso|hipertensa|hypertension|high\s+blood\s+pressure)\b', 'hipertensión'),
            (r'\b(asma|asthma|asmático)\b', 'asma'),
            (r'\b(alergi[ao]s?\s+a|alérgic[oa]\s+a|allergic\s+to)\b', 'alergias'),
            (r'\b(tiroides|hypothyroid|hyperthyroid|tiroiditis)\b', 'tiroides'),
            (r'\b(coraz[oó]n|cardíaco|cardiaco|heart\s+disease|cardiac)\b', 'enfermedad cardíaca'),
            (r'\b(colesterol|cholesterol)\b', 'colesterol'),
            (r'\b(warfarina|warfarin)\b', 'warfarina'),
            (r'\b(metformina|metformin)\b', 'metformina'),
            (r'\b(insulina|insulin)\b', 'insulina'),
        ]

        for msg in user_messages:
            msg_lower = msg.lower()

            if 'name' not in profile:
                for pattern in name_patterns:
                    match = re.search(pattern, msg, re.IGNORECASE)
                    if match:
                        candidate = match.group(1).lower()
                        if candidate not in name_blacklist:
                            profile['name'] = match.group(1).capitalize()
                        break

            if 'age' not in profile:
                for pattern in age_patterns:
                    match = re.search(pattern, msg_lower)
                    if match:
                        age = int(match.group(1))
                        if 1 <= age <= 120:
                            profile['age'] = age
                        break

            for pattern, label in condition_patterns:
                if re.search(pattern, msg_lower):
                    profile.setdefault('conditions', [])
                    if label not in profile['conditions']:
                        profile['conditions'].append(label)

        return profile

    

    def get_context_for_prompt(self, session_id: str, max_messages: int = 4) -> str:
        """
        Returns formatted context ready to inject into LLM prompts.
        EXCLUDES third-party messages (bebé, hijo, madre...)
        """
        if not self.has_context(session_id):
            return ""

        profile = self.extract_user_profile(session_id)
        history = self.get_history(session_id, last_n=max_messages * 2)

        parts = []

        if profile:
            profile_parts = []
            if 'name' in profile:
                profile_parts.append(f"Nombre: {profile['name']}")
            if 'age' in profile:
                profile_parts.append(f"Edad: {profile['age']} años")
            if 'conditions' in profile:
                profile_parts.append(f"Condiciones mencionadas: {', '.join(profile['conditions'])}")
            parts.append("Perfil del usuario:\n" + "\n".join(profile_parts))

        if history:
            turns = []
            count = 0
            for msg in history[:-1]:
                if count >= max_messages:
                    break
                
                if _is_third_party_message(msg["content"]):
                    continue
                role = "Usuario" if msg["role"] == "user" else "Asistente"
                content = msg["content"][:120] + "..." if len(msg["content"]) > 120 else msg["content"]
                turns.append(f"{role}: {content}")
                count += 1
            if turns:
                parts.append("Conversación reciente:\n" + "\n".join(turns))

        return "\n\n".join(parts)

    
    def is_profile_question(self, question: str) -> bool:
        patterns = [
            r'(resúmeme|resume|summarize).{0,20}(perfil|historial|conversaci[oó]n|hablado)',
            r'(recuerdas|recuerda|remember).{0,20}(nombre|edad|condici[oó]n|dijiste|said)',
            r'(qué\s+sé|what\s+do\s+you\s+know).{0,20}(mí|sobre\s+mi|about\s+me)',
            r'(mi\s+perfil|my\s+profile|mon\s+profil)',
            r'(hemos\s+hablado|we\s+talked|nous\s+avons\s+parlé)',
            r'(me\s+puedes\s+resumir|can\s+you\s+summarize)',
            r'(cuál\s+es\s+mi|what\s+is\s+my).{0,20}(nombre|edad|condici[oó]n)',
            r'(c[oó]mo\s+me\s+llamo|what.s\s+my\s+name)',
        ]
        return any(re.search(p, question, re.IGNORECASE) for p in patterns)

    def build_profile_response(self, session_id: str, lang_code: str = 'es') -> str:
        profile = self.extract_user_profile(session_id)
        history = self.get_history(session_id)

        if not profile and not history:
            responses = {
                'es': "No tenemos conversaciones previas. ¿En qué puedo ayudarte?",
                'en': "We don't have any previous conversations. How can I help you?",
                'fr': "Nous n'avons pas de conversations précédentes. Comment puis-je vous aider?",
                'ar': "ليس لدينا محادثات سابقة. كيف يمكنني مساعدتك؟"
            }
            return responses.get(lang_code, responses['es'])

        if lang_code == 'es':
            parts = ["Esto es lo que sé sobre ti a partir de nuestra conversación:"]
            if 'name' in profile:
                parts.append(f"• Nombre: {profile['name']}")
            if 'age' in profile:
                parts.append(f"• Edad: {profile['age']} años")
            if 'conditions' in profile:
                parts.append(f"• Condiciones mencionadas: {', '.join(profile['conditions'])}")
            parts.append(f"• Hemos tenido {len([m for m in history if m['role'] == 'user'])} intercambios en esta sesión.")
            parts.append("\nSi hay algo más que quieras que tenga en cuenta, puedes decírmelo. 💙")

        elif lang_code == 'en':
            parts = ["Here's what I know about you from our conversation:"]
            if 'name' in profile:
                parts.append(f"• Name: {profile['name']}")
            if 'age' in profile:
                parts.append(f"• Age: {profile['age']} years old")
            if 'conditions' in profile:
                parts.append(f"• Mentioned conditions: {', '.join(profile['conditions'])}")
            parts.append(f"• We've had {len([m for m in history if m['role'] == 'user'])} exchanges this session.")
            parts.append("\nFeel free to share anything else you'd like me to keep in mind. 💙")

        elif lang_code == 'fr':
            parts = ["Voici ce que je sais sur vous:"]
            if 'name' in profile:
                parts.append(f"• Nom: {profile['name']}")
            if 'age' in profile:
                parts.append(f"• Âge: {profile['age']} ans")
            if 'conditions' in profile:
                parts.append(f"• Conditions mentionnées: {', '.join(profile['conditions'])}")
            parts.append("\nN'hésitez pas à me dire autre chose. 💙")

        else:
            parts = ["إليك ما أعرفه عنك من محادثتنا:"]
            if 'name' in profile:
                parts.append(f"• الاسم: {profile['name']}")
            if 'age' in profile:
                parts.append(f"• العمر: {profile['age']} سنة")
            if 'conditions' in profile:
                parts.append(f"• الحالات المذكورة: {', '.join(profile['conditions'])}")
            parts.append("\nإذا أردت إضافة أي معلومات أخرى، أخبرني. 💙")

        return "\n".join(parts)

    

    def _save_conversation(self, session_id: str):
        conv_file = self.storage_dir / f"{session_id}.json"
        with open(conv_file, "w", encoding="utf-8") as f:
            json.dump(self.active_conversations[session_id], f, ensure_ascii=False, indent=2)

    def _load_conversation(self, session_id: str):
        conv_file = self.storage_dir / f"{session_id}.json"
        if conv_file.exists():
            with open(conv_file, "r", encoding="utf-8") as f:
                self.active_conversations[session_id] = json.load(f)

    def get_stats(self, session_id: str) -> Dict:
        history = self.get_history(session_id)
        if not history:
            return {"total_messages": 0}
        user_messages = [m for m in history if m["role"] == "user"]
        return {
            "total_messages": len(history),
            "user_messages": len(user_messages),
            "conversation_start": history[0]["timestamp"],
            "last_message": history[-1]["timestamp"],
            "profile": self.extract_user_profile(session_id)
        }