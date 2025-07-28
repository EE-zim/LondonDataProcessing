"""
3GPP/QXDM log processor for sentence and token segmentation.
Goal: minimize sentence length, keep tokens precise and remain extensible.
"""

import re
import spacy
from spacy.tokens import Doc
from typing import Iterator, List, Tuple
from pathlib import Path


class QXDMLogProcessor:
    """Specialized processor for 3GPP/QXDM logs"""
    
    def __init__(self, use_full_model: bool = True):
        """Initialize the log processor

        Args:
            use_full_model: whether to load the full spaCy model (vs blank model)
        """
        self.use_full_model = use_full_model
        self._setup_patterns()
        self._setup_nlp()
    
    def _setup_patterns(self):
        """Configure regular expression patterns"""
        # Record header formats: support multiple timestamp styles
        # Match standard QXDM log header format:
        # 2025 Jul 18  12:11:55.525  [2A]  0x1FEA  Diagnostic Request  --  Message_Type
        # or the legacy format: YYYY Mon DD HH:MM:SS.mmm [XX] 0xXXXX WCDMA/LTE Signaling Messages -- Message_Type
        self.REC_HEADER_RE = re.compile(
            r"^\d{4}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d\d:\d\d:\d\d\.\d{3}\s+\[\w+\]\s+0x\w+\s+.*?--"
        )
        
        # Match technical terms as single tokens
        self.TOKEN_MATCH_RE = re.compile(
            r"""
            0x[0-9A-Fa-f]+ |                              # pointer/HEX prefix
            (?:[0-9A-Fa-f]{2}\s+){3,}[0-9A-Fa-f]{2} |     # long HEX dump
            UL_DCCH|DL_DCCH|DL_BCCH_BCH|UL_CCCH|DL_CCCH | # RRC channels
            DL_BCCH_DL_SCH|UL_DTCH|DL_DTCH |              # additional channels
            BCCH-BCH-Message|UL-DCCH-Message|DL-DCCH-Message | # message types
            MAC-I|MAC-ehs|HARQ |                          # MAC layer
            Uarfcn|Psc|Lac|Rac|CellId |                   # cell parameters
            RRC|MAC|PHY|PDCP|RLC |                        # protocol layers
            SIB\d*|MIB|BCCH |                             # system information
            systemInformationBlockType\d+ |               # SIB type
            masterInformationBlock |                      # MIB
            rrcConnectionRelease|rrcConnectionReleaseComplete | # RRC messages
            rrcConnectionRequest|rrcConnectionSetup |      # RRC connection msgs
            ASN\.1|PDU |                                  # encoding terms
            \d+_\d+ |                                     # like 34_70
            \b\d+\.\d+(?:[eE][+-]?\d+)?\b |              # float/power values
            \b[A-Z]{3,}(?:[_-][A-Z0-9]+)* |              # concatenated keywords
            UMTS|WCDMA|LTE|NR|GSM |                       # standards
            PLMN|IMSI|TMSI|RNTI |                         # identifiers
            integrityCheckInfo|messageAuthenticationCode | # integrity info
            criticalExtensions|NonCriticalExtensions |     # ASN.1 structures
            \b(?:dBm|dB|Hz|kHz|MHz|GHz)\b                 # units
            """,
            re.VERBOSE | re.IGNORECASE
        )
        
        # Patterns for lines that should be filtered out
        self.FILTER_LINES_RE = re.compile(
            r"""
            ^(?:Header|Payload|Length):\s* |                    # raw data lines
            ^\s*Length:\s*\d+\s*$ |                            # length line
            ^\s*Header:\s*[0-9A-Fa-f\s]+$ |                    # header line
            ^\s*Payload:\s*[0-9A-Fa-f\s]+$ |                   # payload line
            ^\s*[0-9A-Fa-f]{2}(?:\s+[0-9A-Fa-f]{2}){3,}\s*$ |  # pure HEX line
            ^\s*HEX\s+Dump\s+of |                              # HEX dump title
            ^\s*\*{5,}.*?\*{5,}\s*$ |                          # separators
            ^\s*RRC\s+Decode:\s+Received |                     # RRC decode info
            ^Subscription\s+ID\s*= |                           # subscription ID
            ^\s*Channel\s+Type\s*= |                           # channel type line
            ^\s*Radio\s+Bearer\s+ID\s*= |                      # radio bearer ID
            ^\s*\|\s*[0-9A-Fa-f\s\|\.]+\s*\|\s*$              # HEX table line
            """,
            re.VERBOSE | re.IGNORECASE
        )
        
        # Sentence boundary keywords - include more 3GPP specific terms
        self.SENTENCE_BOUNDARY_KEYWORDS = {
            "value", "message", "procedure", "event", "indication", 
            "request", "response", "confirm", "primitive", "information",
            "interpreted", "pdu", "rrc", "connection", "release", "complete",
            "mib", "sib", "bcch", "dcch", "ccch", "integrity", "authentication"
        }
    
    def _setup_nlp(self):
        """Set up the spaCy processing pipeline"""
        try:
            if self.use_full_model:
                # Use the full model but exclude unnecessary components for speed
                self.nlp = spacy.load("en_core_web_sm", 
                                    exclude=["ner", "parser", "lemmatizer"])
            else:
                # Use a lightweight blank model
                self.nlp = spacy.blank('en')
                
            # Increase max length to handle large log files (50MB)
            self.nlp.max_length = 50_000_000
            
            # Apply custom token matching rule
            self.nlp.tokenizer.token_match = self.TOKEN_MATCH_RE.match
            
            # Add log boundary detection pipeline before the sentencizer
            @self.nlp.add_pipe("log_boundaries", before="sentencizer")
            def log_boundaries(doc: Doc):
                """Detect log-specific sentence boundaries"""
                for i, tok in enumerate(doc[:-1]):
                    # Braces indicate boundaries
                    if tok.text in {"{", "}"}:
                        doc[i+1].is_sent_start = True
                    # Keyword boundaries
                    elif tok.text.lower() in self.SENTENCE_BOUNDARY_KEYWORDS:
                        if i > 0:  # keyword itself starts new sentence
                            doc[i].is_sent_start = True
                    # Double newline boundary
                    elif "\n\n" in tok.text:
                        doc[i+1].is_sent_start = True
                return doc
            
            # Add the sentencizer
            self.nlp.add_pipe("sentencizer", config={
                'punct_chars': ['.', '?', '!', '\n', ';', ':', '|']
            })
            
        except OSError:
            # Fallback to blank model if en_core_web_sm is unavailable
            print("[warn] en_core_web_sm not available, using blank model")
            self.nlp = spacy.blank('en')
            self.nlp.max_length = 50_000_000  # same increased limit for blank model
            self.nlp.tokenizer.token_match = self.TOKEN_MATCH_RE.match
            self.nlp.add_pipe('sentencizer')
    
    def iterate_records(self, raw_text: str) -> Iterator[List[str]]:
        """Iterate over log records and yield lists of lines

        Args:
            raw_text: raw log text

        Yields:
            list of lines for a single log record
        """
        buf = []
        for line in raw_text.splitlines(keepends=False):
            # Detect start of a new record
            if self.REC_HEADER_RE.match(line):
                if buf:
                    yield buf
                    buf.clear()
            buf.append(line.rstrip())
        
        # Handle last record
        if buf:
            yield buf
    
    def _should_skip_line(self, line: str) -> bool:
        """Return True if the line should be skipped (mostly HEX dumps)"""
        line_stripped = line.strip()
        
        # Skip lines that are raw HEX dumps
        if re.match(r'^\s*(?:Length|Header|Payload):\s*[0-9A-Fa-f\s]+$', line_stripped):
            return True
            
        # Skip lines consisting purely of HEX data
        if re.match(r'^\s*[0-9A-Fa-f]{2}(?:\s+[0-9A-Fa-f]{2}){3,}\s*$', line_stripped):
            return True
            
        return False
    
    def _extract_rrc_content(self, record_text: str) -> str:
        """Extract RRC related technical content from a record

        Args:
            record_text: full record text

        Returns:
            extracted technical content
        """
        lines = record_text.splitlines()
        result_lines = []
        in_asn1 = False
        asn1_brace_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Always keep the record header (timestamp line)
            if self.REC_HEADER_RE.match(line):
                result_lines.append(line)
                continue
            
            # Detect start of ASN.1 structure
            if "Interpreted PDU:" in line or ("value " in line and "::=" in line):
                in_asn1 = True
                asn1_brace_count = 0
                result_lines.append(line)
                continue
            
            # If inside an ASN.1 block, keep all content
            if in_asn1:
                result_lines.append(line)
                asn1_brace_count += line.count('{') - line.count('}')
                if asn1_brace_count == 0 and '}' in line:
                    in_asn1 = False
                continue
            
            # Skip HEX dump lines
            if self._should_skip_line(line):
                continue
            
            # Keep lines with important technical info
            if any(keyword in line_stripped.lower() for keyword in [
                "subscription id", "radio bearer id", "physical cell id", 
                "freq =", "sysframenum", "subframenum", "pdu number",
                "msg length", "sib mask", "rrc release number",
                "pkt version", "cmd code", "subsys id", "subsys cmd code"
            ]):
                result_lines.append(line)
            
            # Keep other descriptive lines (non HEX)
            elif not self._is_pure_hex_line(line_stripped) and len(line_stripped) > 5:
                # Check if line contains technical content
                if any(tech_word in line_stripped.lower() for tech_word in [
                    "rrc", "mac", "phy", "pdcp", "rlc", "message", "procedure",
                    "lte", "wcdma", "umts", "signaling", "bearer", "cell"
                ]):
                    result_lines.append(line)
        
        return "\n".join(result_lines)

    def _is_asn1_structure_line(self, line: str) -> bool:
        """Check whether a line is part of an ASN.1 structure"""
        # ASN.1 indicators: braces, colons and special keywords
        asn1_indicators = ['{', '}', '::=', 'value', 'message', 'criticalExtensions',
                          'dl-CarrierFreq', 'q-RxLevMin', 't-ReselectionEUTRA',
                          'threshX-', 'allowedMeasBandwidth', 'cellReselectionPriority',
                          'sf-Medium', 'sf-High', 'presenceAntennaPort', 'neighCellConfig']
        
        return any(indicator in line for indicator in asn1_indicators)
    
    def _is_pure_hex_line(self, line: str) -> bool:
        """Return True if the line is purely HEX data"""
        # Remove spaces and common separators
        cleaned = re.sub(r'[\s\|\-\.]', '', line)
        if len(cleaned) < 8:  # too short to be considered HEX
            return False
        
        # Check if all characters are HEX
        return all(c in "0123456789ABCDEFabcdef" for c in cleaned)
    
    def _is_hex_token(self, token: str) -> bool:
        """Check whether a token is a long HEX string"""
        if len(token) <= 4:
            return False
        
        # Remove 0x prefix
        hex_part = token[2:] if token.startswith('0x') else token
        
        # Check if token is long HEX (>4 chars and all HEX)
        return (len(hex_part) > 4 and 
                all(c in "0123456789ABCDEFabcdef" for c in hex_part))
    
    def clean_sentence(self, sent) -> Tuple[str, bool]:
        """Clean a single sentence

        Args:
            sent: spaCy Span object

        Returns:
            (cleaned sentence text, keep_flag)
        """
        # Filter out long HEX tokens
        tokens = []
        for token in sent:
            if not self._is_hex_token(token.text):
                tokens.append(token.text)
        
        if len(tokens) < 3:  # too short
            return "", False
        
        sent_text = " ".join(tokens)
        
        # Length check - increased limit to keep ASN.1 structures
        if len(sent_text) > 2000:  # was 300 characters
            return "", False
        
        # Check if it is a meaningful technical sentence
        if self._has_technical_content(sent_text):
            return sent_text.strip(), True
        
        return "", False
    
    def _has_technical_content(self, text: str) -> bool:
        """Check whether text contains technical content"""
        text_lower = text.lower()
        
        # Contains technical keywords
        technical_keywords = [
            'rrc', 'mac', 'phy', 'pdcp', 'rlc', 'pdu', 'message',
            'procedure', 'event', 'value', 'asn.1', 'decoded',
            'uarfcn', 'psc', 'cell', 'frequency', 'power', 'sib', 'mib',
            'bcch', 'dcch', 'ccch', 'carrierfreq', 'reselection',
            'threshold', 'bandwidth', 'priority', 'antenna', 'config'
        ]
        
        if any(keyword in text_lower for keyword in technical_keywords):
            return True
        
        # ASN.1 structural cues
        if any(pattern in text for pattern in ['::=', '{', '}', 'dl-', 'ul-', 'sf-']):
            return True
        
        # Contains numeric values and units
        if re.search(r'\d+\s*(?:db|hz|khz|mhz|ghz|ms|sec)', text_lower):
            return True
        
        # Contains protocol fields
        if re.search(r'[a-z]+_[a-z]+|[A-Z]{2,}', text):
            return True
        
        return False
    
    def _extract_clean_lines(self, rec: List[str]) -> List[str]:
        """Extract clean, meaningful sentences/blocks from a raw record.
        1. Pull the entire ASN.1 block into a single string.
        2. Collect remaining descriptive lines into another string.
        3. Filter out all HEX dump lines.
        """
        descriptive_lines = []
        asn1_blocks = []
        i = 0
        
        # Regexes for pure HEX lines and common prefixes
        hex_line_re = re.compile(r"^\s*((Length:|Header:|Payload:)\s*)?([0-9A-Fa-f]{2}\s*)+\s*$")
        hex_prefix_re = re.compile(r"^\s*(Length:|Header:|Payload:|HEX Dump)")

        while i < len(rec):
            line = rec[i]

            # Skip anything that looks like a HEX dump or its header
            if hex_prefix_re.match(line) or hex_line_re.match(line):
                i += 1
                continue

            # Encounter an "Interpreted PDU" block
            if line.strip().startswith("Interpreted PDU:"):
                j = i + 1
                # Look for the start of the 'value' block
                while j < len(rec) and not rec[j].lstrip().startswith("value"):
                    # Lines between "Interpreted PDU" and "value" are descriptive
                    if rec[j].strip():
                        descriptive_lines.append(rec[j])
                    j += 1
                
                if j < len(rec):
                    # Found 'value', now capture the whole block
                    brace_depth = 0
                    rrc_block_lines = []
                    k = j
                    while k < len(rec):
                        r_line = rec[k]
                        brace_depth += r_line.count("{") - r_line.count("}")
                        rrc_block_lines.append(r_line.strip())
                        if brace_depth == 0 and '}' in r_line:
                            # Ensure we stop when braces match
                            if r_line.strip().endswith("}"):
                                break
                        k += 1
                    
                    asn1_blocks.append(" ".join(rrc_block_lines))
                    i = k + 1
                else:
                    # 'value' not found; continue
                    i = j
                continue

            # Ordinary descriptive line
            if line.strip():
                descriptive_lines.append(line)
            i += 1

        # Combine results
        final_cleaned_lines = []
        if descriptive_lines:
            final_cleaned_lines.append(" ".join(descriptive_lines))
        final_cleaned_lines.extend(asn1_blocks)
        
        return final_cleaned_lines

    def _split_sentences(self, lines: List[str]) -> List[str]:
        """Run cleaned lines/blocks through spaCy's sentencizer.
        ASN.1 blocks are passed through unchanged.
        """
        sents = []
        for ln in lines:
            # Pre-parsed ASN.1 blocks are not further split
            if ln.lstrip().startswith("value") and "::=" in ln:
                sents.append(ln)
            else:
                # Descriptive block; let spaCy split as needed
                doc = self.nlp(ln)
                sents.extend(sent.text.strip() for sent in doc.sents if sent.text.strip())
        return sents

    def process_file(self, file_path: Path) -> List[str]:
        """Process a single log file

        Args:
            file_path: path to the log file

        Returns:
            list of extracted sentences
        """
        try:
            # Read the file
            if file_path.suffix.lower() == '.pdf':
                try:
                    from pdfminer.high_level import extract_text
                    raw_text = extract_text(str(file_path))
                except ImportError:
                    print(f"[warn] Cannot process PDF {file_path.name}: pdfminer.six not installed")
                    return []
            else:
                raw_text = file_path.read_text(encoding='utf-8', errors='ignore')
            
            all_sentences = []
            # Process record by record
            for record in self.iterate_records(raw_text):
                clean_lines = self._extract_clean_lines(record)
                sentences = self._split_sentences(clean_lines)
                all_sentences.extend(sentences)
            
            return all_sentences
            
        except Exception as e:
            print(f"[warn] Error reading {file_path}: {e}")
            return []
    
    def get_stats(self) -> dict:
        """Return processor statistics"""
        return {
            'model_type': 'full' if self.use_full_model else 'blank',
            'pipeline_components': [comp for comp in self.nlp.pipe_names],
            'max_length': self.nlp.max_length,
            'patterns_count': {
                'record_header': 1,
                'token_match': 1,
                'filter_lines': 1,
                'sentence_boundaries': len(self.SENTENCE_BOUNDARY_KEYWORDS)
            }
        }


def test_processor():
    """Test processor functionality"""
    # Create test data
    test_log = """2024 Jul 28 14:30:25.123 [RRC] UL_DCCH
Interpreted PDU:
RRCConnectionRequest ::= {
    message c1 : rrcConnectionRequest : {
        criticalExtensions rrcConnectionRequest-r8 : {
            ue-Identity randomValue : '0110 1001 0101 1010'B
            establishmentCause mt-Access
        }
    }
}

Header: 0x412F
Payload: A1 B2 C3 D4 E5 F6 
Length: 24 bytes

2024 Jul 28 14:30:26.456 [MAC] HARQ Process 3
Value: Uarfcn = 2850, Psc = 123
BLER: 0.01 dB
"""
    
    processor = QXDMLogProcessor()
    
    # Test record splitting
    records = list(processor.iterate_records(test_log))
    print(f"Found {len(records)} records")
    
    # Test sentence extraction
    all_sentences = []
    for filtered_record in processor.iterate_records(test_log):
        if filtered_record.strip():
            doc = processor.nlp(filtered_record)
            for sent in doc.sents:
                clean_text, keep = processor.clean_sentence(sent)
                if keep:
                    all_sentences.append(clean_text)
    
    print(f"Extracted {len(all_sentences)} sentences:")
    for i, sent in enumerate(all_sentences, 1):
        print(f"  {i}. {sent}")
    
    # Statistics
    stats = processor.get_stats()
    print(f"\nProcessor stats: {stats}")


if __name__ == "__main__":
    test_processor()
