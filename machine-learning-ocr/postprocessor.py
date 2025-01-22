import re
from typing import Dict, List, Tuple, Optional

class CharacterSimilarity:
    """Character similarity mappings for OCR correction"""
    # Similar looking characters (bidirectional mapping)
    SIMILAR_CHARS = {
        'B': '8',
        'D': '0',
        'O': '0',
        'Q': '0',
        'S': '5',
        'Z': '2',
        'I': '1',
        'A': '4',
        'G': '6',
        'T': '7',
        'U': 'V',
        'V': 'U',
    }
    
    # Create reverse mappings
    REVERSE_SIMILAR = {v: k for k, v in SIMILAR_CHARS.items()}
    
    @classmethod
    def get_similar_letter(cls, digit: str) -> Optional[str]:
        """Convert a digit to a similar looking letter"""
        return cls.REVERSE_SIMILAR.get(digit)
    
    @classmethod
    def get_similar_digit(cls, letter: str) -> Optional[str]:
        """Convert a letter to a similar looking digit"""
        return cls.SIMILAR_CHARS.get(letter)

class LicensePlatePostProcessor:
    """Post-processor for Brazilian license plates"""
    
    # Regular expressions for valid Brazilian plate formats
    OLD_FORMAT = re.compile(r'^[A-Z]{3}[0-9]{4}$')  # AAA9999
    NEW_FORMAT = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')  # AAA9A99
    
    def __init__(self):
        self.similarity = CharacterSimilarity()
    
    def is_valid_plate(self, plate: str) -> bool:
        """Check if a plate matches either format"""
        return bool(self.OLD_FORMAT.match(plate) or self.NEW_FORMAT.match(plate))
    
    def clean_text(self, text: str) -> str:
        """Clean text from common OCR artifacts"""
        # Remove common separators and spaces
        text = re.sub(r'[-_.\s]', '', text)
        # Remove any non-alphanumeric characters
        text = re.sub(r'[^A-Z0-9]', '', text.upper())
        return text
    
    def try_fix_position(self, text: str, position: int, expected_type: str) -> Tuple[str, bool]:
        """
        Try to fix a character at a specific position based on expected type.
        Args:
            text: The plate text
            position: Position to fix (0-based)
            expected_type: Either 'letter' or 'digit'
        Returns:
            Tuple of (fixed text, whether it was fixed)
        """
        if position >= len(text):
            return text, False
            
        char = text[position]
        fixed = text
        
        if expected_type == 'letter' and char.isdigit():
            similar_letter = self.similarity.get_similar_letter(char)
            if similar_letter:
                fixed = text[:position] + similar_letter + text[position + 1:]
                return fixed, True
                
        elif expected_type == 'digit' and char.isalpha():
            similar_digit = self.similarity.get_similar_digit(char)
            if similar_digit:
                fixed = text[:position] + similar_digit + text[position + 1:]
                return fixed, True
                
        return text, False
    
    def extract_potential_plate(self, text: str) -> str:
        """Extract potential plate from longer text"""
        # Look for 7-character sequences that could be a plate
        matches = re.finditer(r'[A-Z0-9]{7}', text)
        for match in matches:
            candidate = match.group()
            # Try both formats
            if self.try_process_format(candidate):
                return candidate
        return text
    
    def try_process_format(self, text: str) -> Optional[str]:
        """Try to process text in a specific format"""
        if len(text) != 7:
            return None
            
        # Try old format (AAA9999)
        fixed = text
        for i in range(7):
            expected_type = 'letter' if i < 3 else 'digit'
            fixed, _ = self.try_fix_position(fixed, i, expected_type)
        if self.is_valid_plate(fixed):
            return fixed
            
        # Try new format (AAA9A99)
        fixed = text
        for i in range(7):
            if i < 3:  # First three are letters
                expected_type = 'letter'
            elif i == 3:  # Fourth is digit
                expected_type = 'digit'
            elif i == 4:  # Fifth is letter
                expected_type = 'letter'
            else:  # Last two are digits
                expected_type = 'digit'
            fixed, _ = self.try_fix_position(fixed, i, expected_type)
        if self.is_valid_plate(fixed):
            return fixed
            
        return None
    
    def process(self, text: str) -> str:
        """
        Process and correct a license plate text.
        Args:
            text: Raw OCR text
        Returns:
            Corrected license plate text or empty string if unfixable
        """
        # Clean the input
        text = self.clean_text(text)
        
        # If empty after cleaning, return empty
        if not text:
            return ""
            
        # If already valid, return as is
        if self.is_valid_plate(text):
            return text
            
        # If text is longer than 7 chars, try to extract plate
        if len(text) > 7:
            text = self.extract_potential_plate(text)
        
        # If text is shorter than 7 chars, can't be a plate
        if len(text) < 7:
            return ""
            
        # Try to process in both formats
        result = self.try_process_format(text)
        if result:
            return result
        
        return ""  # Return empty string if we couldn't fix it 