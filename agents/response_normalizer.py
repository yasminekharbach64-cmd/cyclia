"""
Response Normalizer Agent
Ensures all responses maintain appropriate medical safety standards
Reduces overconfidence, adds disclaimers, prevents diagnostic language
"""

import re
from typing import Dict, Any
from agents.risk_agent import RiskLevel


class ResponseNormalizer:
    """
    Validates and normalizes health responses to ensure medical safety
    This is the CONTROL layer that prevents overconfident or dangerous responses
    """
    
    def __init__(self):
        
        self.forbidden_diagnostic_words = {
            'es': ['tienes', 'sufres', 'padeces', 'diagnóstico', 'es probable que tengas'],
            'en': ['you have', 'you suffer from', 'you are diagnosed', 'diagnosis'],
            'fr': ['tu as', 'vous avez', 'tu souffres', 'diagnostic']
        }
        
        
        self.overconfident_patterns = {
            'es': [
                (r'\bestá relacionado con\b', 'podría estar relacionado con'),
                (r'\bes causado por\b', 'puede ser causado por'),
                (r'\bsufres de\b', 'podrías tener síntomas de'),
                (r'\bsí,\s*', ''),  
                (r'\bestás\s+\w+\s+de\b', 'podrías estar experimentando'),
            ],
            'en': [
                (r'\bis caused by\b', 'may be caused by'),
                (r'\byou have\b', 'you might be experiencing'),
                (r'\byes,\s*', ''),  
                (r'\byou are suffering\b', 'you may be experiencing'),
            ],
            'fr': [
                (r'\best causé par\b', 'peut être causé par'),
                (r'\btu as\b', 'tu pourrais avoir'),
                (r'\boui,\s*', ''),  
                (r'\bvous avez\b', 'vous pourriez avoir'),
            ]
        }
    
    def normalize(self, response: str, risk_level: RiskLevel, language: str) -> str:
        """
        Main normalization function
        
        Args:
            response: Original response from health agent
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            language: Language code (es/en/fr)
        
        Returns:
            Normalized, safer response
        """
        
        
        if self._contains_diagnostic_language(response, language):
            return self._get_safe_referral(language, risk_level)
        
       
        response = self._reduce_overconfidence(response, language)
        
        
        if risk_level == RiskLevel.HIGH:
            response = self._normalize_high_risk(response, language)
        elif risk_level == RiskLevel.MEDIUM:
            response = self._normalize_medium_risk(response, language)
        else:  
            response = self._normalize_low_risk(response, language)
        
        
        response = self._ensure_proper_structure(response, risk_level, language)
        
        return response
    
    def _contains_diagnostic_language(self, response: str, language: str) -> bool:
        """Check if response contains forbidden diagnostic language"""
        forbidden = self.forbidden_diagnostic_words.get(language, [])
        response_lower = response.lower()
        
        for word in forbidden:
            if word in response_lower:
                return True
        
        
        diagnostic_patterns = [
            r'^sí,\s+.*covid',
            r'^yes,\s+.*covid',
            r'^oui,\s+.*covid',
            r'\btienes\s+(covid|cáncer|diabetes)',
            r'\byou have\s+(covid|cancer|diabetes)',
        ]
        
        for pattern in diagnostic_patterns:
            if re.search(pattern, response_lower, re.IGNORECASE):
                return True
        
        return False
    
    def _reduce_overconfidence(self, response: str, language: str) -> str:
        """Reduce certainty in response"""
        patterns = self.overconfident_patterns.get(language, [])
        
        for pattern, replacement in patterns:
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        
        return response
    
    def _normalize_high_risk(self, response: str, language: str) -> str:
        """
        HIGH RISK: Maximum caution, minimal detail, clear referral
        Should be very brief and redirect to professional
        """
       
        sentences = response.split('.')
        
        
        if len(sentences) > 3:
            response = '. '.join(sentences[:3]) + '.'
        
        
        referral_phrases = {
            'es': 'Es fundamental consultar con un profesional sanitario para una evaluación adecuada.',
            'en': 'It is essential to consult with a healthcare professional for proper evaluation.',
            'fr': 'Il est essentiel de consulter un professionnel de santé pour une évaluation appropriée.'
        }
        
        if 'médico' not in response.lower() and 'doctor' not in response.lower():
            response += ' ' + referral_phrases.get(language, referral_phrases['es'])
        
        return response
    
    def _normalize_medium_risk(self, response: str, language: str) -> str:
        """
        MEDIUM RISK: Conditional language, acknowledge uncertainty
        """
        
        conditional_intros = {
            'es': 'Sin poder determinar una causa específica, ',
            'en': 'Without being able to determine a specific cause, ',
            'fr': 'Sans pouvoir déterminer une cause spécifique, '
        }
        
        
        has_conditional = any(phrase in response.lower() for phrase in 
                            ['podría', 'puede ser', 'posible', 'might', 'could', 'possible', 'peut-être'])
        
        if not has_conditional:
            intro = conditional_intros.get(language, conditional_intros['es'])
            
            sentences = response.split('. ')
            if len(sentences) > 1:
                sentences[0] = intro + sentences[0].lower()
                response = '. '.join(sentences)
        
        return response
    
    def _normalize_low_risk(self, response: str, language: str) -> str:
        """
        LOW RISK: Can be more informative but still general
        """
        
        educational_phrases = {
            'es': 'información general',
            'en': 'general information',
            'fr': 'information générale'
        }
        
        
        return response
    
    def _ensure_proper_structure(self, response: str, risk_level: RiskLevel, language: str) -> str:
        """Ensure response has proper structure and length"""
        
        
        response = ' '.join(response.split())
        
        
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        
        response = self._add_disclaimer(response, risk_level, language)
        
        return response
    
    def _add_disclaimer(self, response: str, risk_level: RiskLevel, language: str) -> str:
        """Add appropriate medical disclaimer based on risk level"""
        
        
        has_disclaimer = any(word in response.lower() for word in 
                           ['médico', 'doctor', 'professionnel', 'profesional'])
        
        if has_disclaimer:
            return response
        
        disclaimers = {
            RiskLevel.HIGH: {
                'es': ' Consulta con un profesional sanitario de inmediato.',
                'en': ' Consult with a healthcare professional immediately.',
                'fr': ' Consultez un professionnel de santé immédiatement.'
            },
            RiskLevel.MEDIUM: {
                'es': ' Si los síntomas persisten o empeoran, consulta con un médico.',
                'en': ' If symptoms persist or worsen, consult with a doctor.',
                'fr': ' Si les symptômes persistent ou s\'aggravent, consultez un médecin.'
            },
            RiskLevel.LOW: {
                'es': '',  
                'en': '',
                'fr': ''
            }
        }
        
        disclaimer = disclaimers[risk_level].get(language, disclaimers[risk_level]['es'])
        return response + disclaimer
    
    def _get_safe_referral(self, language: str, risk_level: RiskLevel) -> str:
        """Return safe referral response when diagnostic language detected"""
        referrals = {
            'es': 'No puedo proporcionar diagnósticos específicos. Los síntomas que describes requieren una evaluación médica profesional para determinar su causa. Por favor, consulta con un profesional sanitario. 💙',
            'en': 'I cannot provide specific diagnoses. The symptoms you describe require professional medical evaluation to determine their cause. Please consult with a healthcare professional. 💙',
            'fr': 'Je ne peux pas fournir de diagnostics spécifiques. Les symptômes que vous décrivez nécessitent une évaluation médicale professionnelle. Veuillez consulter un professionnel de santé. 💙'
        }
        return referrals.get(language, referrals['es'])
    
    def validate_response_quality(self, response: str, risk_level: RiskLevel) -> Dict[str, Any]:
        """
        Validate response quality and safety
        Returns metrics for logging
        """
        
        metrics = {
            'length': len(response),
            'sentence_count': len(response.split('.')),
            'has_disclaimer': any(word in response.lower() for word in 
                                ['médico', 'doctor', 'professionnel']),
            'certainty_level': self._assess_certainty(response),
            'is_safe': True
        }
        
        
        if risk_level == RiskLevel.HIGH and metrics['sentence_count'] > 4:
            metrics['is_safe'] = False
            metrics['issue'] = 'HIGH risk response too detailed'
        
        if metrics['certainty_level'] == 'high' and risk_level != RiskLevel.LOW:
            metrics['is_safe'] = False
            metrics['issue'] = 'Overconfident language detected'
        
        return metrics
    
    def _assess_certainty(self, response: str) -> str:
        """Assess certainty level of response"""
        high_certainty = ['es', 'está', 'tienes', 'sufres', 'is', 'you have', 'certainly']
        medium_certainty = ['podría', 'puede', 'posible', 'might', 'could', 'possible']
        
        response_lower = response.lower()
        
        high_count = sum(1 for word in high_certainty if word in response_lower)
        medium_count = sum(1 for word in medium_certainty if word in response_lower)
        
        if high_count > medium_count:
            return 'high'
        elif medium_count > 0:
            return 'medium'
        else:
            return 'low'



if __name__ == "__main__":
    normalizer = ResponseNormalizer()
    
    print("=" * 70)
    print("RESPONSE NORMALIZER - TESTING")
    print("=" * 70)
    
    test_cases = [
        (
            "Sí, la fiebre es uno de los síntomas comunes del COVID-19. Si tienes fiebre...",
            RiskLevel.HIGH,
            "es",
            "Should remove 'Sí' and make less diagnostic"
        ),
        (
            "El dolor está relacionado con el estrés. Sufres de tensión muscular.",
            RiskLevel.MEDIUM,
            "es",
            "Should reduce certainty and add conditionals"
        ),
        (
            "Para dormir mejor, mantén un horario regular.",
            RiskLevel.LOW,
            "es",
            "Should remain mostly unchanged"
        ),
    ]
    
    for original, risk, lang, description in test_cases:
        print(f"\n{'='*70}")
        print(f"[{description}]")
        print(f"Risk: {risk.value}")
        print(f"{'='*70}")
        print(f"ORIGINAL:\n{original}")
        
        normalized = normalizer.normalize(original, risk, lang)
        print(f"\nNORMALIZED:\n{normalized}")
        
        quality = normalizer.validate_response_quality(normalized, risk)
        print(f"\nQUALITY METRICS:")
        print(f"  Certainty: {quality['certainty_level']}")
        print(f"  Safe: {quality['is_safe']}")
        print(f"  Sentences: {quality['sentence_count']}")
        print("-" * 70)