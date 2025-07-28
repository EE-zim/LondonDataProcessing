"""
3GPP/QXDM 信令日志专用分句与分词处理器
设计目标：句最小化 + 词精准化 + 可扩展性
"""

import re
import spacy
from spacy.tokens import Doc
from typing import Iterator, List, Tuple
from pathlib import Path


class QXDMLogProcessor:
    """3GPP/QXDM 日志专用处理器"""
    
    def __init__(self, use_full_model: bool = True):
        """
        初始化日志处理器
        
        Args:
            use_full_model: 是否使用完整spaCy模型（vs blank模型）
        """
        self.use_full_model = use_full_model
        self._setup_patterns()
        self._setup_nlp()
    
    def _setup_patterns(self):
        """设置正则模式"""
        # 记录首行模式：支持多种时间戳格式
        # 匹配标准QXDM日志记录头格式：
        # 2025 Jul 18  12:11:55.525  [2A]  0x1FEA  Diagnostic Request  --  Message_Type
        # 或原格式：YYYY Mon DD HH:MM:SS.mmm [XX] 0xXXXX WCDMA/LTE Signaling Messages -- Message_Type
        self.REC_HEADER_RE = re.compile(
            r"^\d{4}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d\d:\d\d:\d\d\.\d{3}\s+\[\w+\]\s+0x\w+\s+.*?--"
        )
        
        # 技术字段整体匹配，保持为原子词
        self.TOKEN_MATCH_RE = re.compile(
            r"""
            0x[0-9A-Fa-f]+ |                              # 指针/HEX 前缀
            (?:[0-9A-Fa-f]{2}\s+){3,}[0-9A-Fa-f]{2} |     # 长 HEX dump
            UL_DCCH|DL_DCCH|DL_BCCH_BCH|UL_CCCH|DL_CCCH | # RRC 信道
            DL_BCCH_DL_SCH|UL_DTCH|DL_DTCH |              # 更多信道类型
            BCCH-BCH-Message|UL-DCCH-Message|DL-DCCH-Message | # 消息类型
            MAC-I|MAC-ehs|HARQ |                          # MAC 层
            Uarfcn|Psc|Lac|Rac|CellId |                   # 小区参数
            RRC|MAC|PHY|PDCP|RLC |                        # 协议层
            SIB\d*|MIB|BCCH |                             # 系统信息
            systemInformationBlockType\d+ |               # SIB类型
            masterInformationBlock |                      # MIB
            rrcConnectionRelease|rrcConnectionReleaseComplete | # RRC消息
            rrcConnectionRequest|rrcConnectionSetup |      # RRC连接消息
            ASN\.1|PDU |                                  # 编码相关
            \d+_\d+ |                                     # 像 34_70
            \b\d+\.\d+(?:[eE][+-]?\d+)?\b |              # 浮点/功率值
            \b[A-Z]{3,}(?:[_-][A-Z0-9]+)* |              # 连写大写关键字
            UMTS|WCDMA|LTE|NR|GSM |                       # 技术标准
            PLMN|IMSI|TMSI|RNTI |                         # 标识符
            integrityCheckInfo|messageAuthenticationCode | # 完整性相关
            criticalExtensions|NonCriticalExtensions |     # ASN.1结构
            \b(?:dBm|dB|Hz|kHz|MHz|GHz)\b                 # 单位
            """,
            re.VERBOSE | re.IGNORECASE
        )
        
        # 需要过滤的行模式 - 更精确地匹配真实日志格式
        self.FILTER_LINES_RE = re.compile(
            r"""
            ^(?:Header|Payload|Length):\s* |                    # 原始数据行
            ^\s*Length:\s*\d+\s*$ |                            # 长度行
            ^\s*Header:\s*[0-9A-Fa-f\s]+$ |                    # Header行  
            ^\s*Payload:\s*[0-9A-Fa-f\s]+$ |                   # Payload行
            ^\s*[0-9A-Fa-f]{2}(?:\s+[0-9A-Fa-f]{2}){3,}\s*$ |  # 纯HEX行
            ^\s*HEX\s+Dump\s+of |                              # HEX Dump标题
            ^\s*\*{5,}.*?\*{5,}\s*$ |                          # 分隔线
            ^\s*RRC\s+Decode:\s+Received |                     # RRC解码信息
            ^Subscription\s+ID\s*= |                           # 订阅ID行
            ^\s*Channel\s+Type\s*= |                           # 信道类型行（单独一行时）
            ^\s*Radio\s+Bearer\s+ID\s*= |                      # 无线承载ID
            ^\s*\|\s*[0-9A-Fa-f\s\|\.]+\s*\|\s*$              # HEX表格行
            """,
            re.VERBOSE | re.IGNORECASE
        )
        
        # 句边界关键字 - 添加更多3GPP特定关键字
        self.SENTENCE_BOUNDARY_KEYWORDS = {
            "value", "message", "procedure", "event", "indication", 
            "request", "response", "confirm", "primitive", "information",
            "interpreted", "pdu", "rrc", "connection", "release", "complete",
            "mib", "sib", "bcch", "dcch", "ccch", "integrity", "authentication"
        }
    
    def _setup_nlp(self):
        """设置spaCy处理管线"""
        try:
            if self.use_full_model:
                # 使用完整模型，但排除不需要的组件以提升性能
                self.nlp = spacy.load("en_core_web_sm", 
                                    exclude=["ner", "parser", "lemmatizer"])
            else:
                # 使用轻量blank模型
                self.nlp = spacy.blank('en')
                
            # 设置最大长度限制 - 增加到50MB以处理大型日志文件
            self.nlp.max_length = 50_000_000
            
            # 设置自定义token匹配规则
            self.nlp.tokenizer.token_match = self.TOKEN_MATCH_RE.match
            
            # 添加日志边界检测管线（在sentencizer之前）
            @self.nlp.add_pipe("log_boundaries", before="sentencizer")
            def log_boundaries(doc: Doc):
                """检测日志特定的句边界"""
                for i, tok in enumerate(doc[:-1]):
                    # 花括号边界
                    if tok.text in {"{", "}"}:
                        doc[i+1].is_sent_start = True
                    # 关键字边界
                    elif tok.text.lower() in self.SENTENCE_BOUNDARY_KEYWORDS:
                        if i > 0:  # 关键字本身开始新句
                            doc[i].is_sent_start = True
                    # 双换行边界
                    elif "\n\n" in tok.text:
                        doc[i+1].is_sent_start = True
                return doc
            
            # 添加sentencizer
            self.nlp.add_pipe("sentencizer", config={
                'punct_chars': ['.', '?', '!', '\n', ';', ':', '|']
            })
            
        except OSError:
            # 如果en_core_web_sm不可用，降级到blank模型
            print("[warn] en_core_web_sm not available, using blank model")
            self.nlp = spacy.blank('en')
            self.nlp.max_length = 50_000_000  # 同样增加blank模型的限制
            self.nlp.tokenizer.token_match = self.TOKEN_MATCH_RE.match
            self.nlp.add_pipe('sentencizer')
    
    def iterate_records(self, raw_text: str) -> Iterator[List[str]]:
        """
        按记录级别迭代日志内容，返回记录行列表
        
        Args:
            raw_text: 原始日志文本
            
        Yields:
            单条日志记录的行列表
        """
        buf = []
        for line in raw_text.splitlines(keepends=False):
            # 检测新记录开始
            if self.REC_HEADER_RE.match(line):
                if buf:
                    yield buf
                    buf.clear()
            buf.append(line.rstrip())
        
        # 处理最后一条记录
        if buf:
            yield buf
    
    def _should_skip_line(self, line: str) -> bool:
        """判断是否应该跳过某一行（主要是HEX dump）"""
        line_stripped = line.strip()
        
        # 跳过纯HEX dump行
        if re.match(r'^\s*(?:Length|Header|Payload):\s*[0-9A-Fa-f\s]+$', line_stripped):
            return True
            
        # 跳过纯HEX数据行
        if re.match(r'^\s*[0-9A-Fa-f]{2}(?:\s+[0-9A-Fa-f]{2}){3,}\s*$', line_stripped):
            return True
            
        return False
    
    def _extract_rrc_content(self, record_text: str) -> str:
        """
        从记录中提取RRC相关的技术内容
        
        Args:
            record_text: 完整记录文本
            
        Returns:
            提取的技术内容
        """
        lines = record_text.splitlines()
        result_lines = []
        in_asn1 = False
        asn1_brace_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # 始终保留记录头（时间戳行）
            if self.REC_HEADER_RE.match(line):
                result_lines.append(line)
                continue
            
            # 检测ASN.1结构开始
            if "Interpreted PDU:" in line or ("value " in line and "::=" in line):
                in_asn1 = True
                asn1_brace_count = 0
                result_lines.append(line)
                continue
            
            # 如果在ASN.1结构内，保留所有内容
            if in_asn1:
                result_lines.append(line)
                asn1_brace_count += line.count('{') - line.count('}')
                if asn1_brace_count == 0 and '}' in line:
                    in_asn1 = False
                continue
            
            # 跳过HEX dump行
            if self._should_skip_line(line):
                continue
            
            # 保留包含重要技术信息的行
            if any(keyword in line_stripped.lower() for keyword in [
                "subscription id", "radio bearer id", "physical cell id", 
                "freq =", "sysframenum", "subframenum", "pdu number",
                "msg length", "sib mask", "rrc release number",
                "pkt version", "cmd code", "subsys id", "subsys cmd code"
            ]):
                result_lines.append(line)
            
            # 保留其他技术描述行（非HEX的）
            elif not self._is_pure_hex_line(line_stripped) and len(line_stripped) > 5:
                # 检查是否包含技术内容
                if any(tech_word in line_stripped.lower() for tech_word in [
                    "rrc", "mac", "phy", "pdcp", "rlc", "message", "procedure",
                    "lte", "wcdma", "umts", "signaling", "bearer", "cell"
                ]):
                    result_lines.append(line)
        
        return "\n".join(result_lines)

    def _is_asn1_structure_line(self, line: str) -> bool:
        """检查是否为ASN.1结构行"""
        # ASN.1结构特征：大括号、冒号、特殊关键字
        asn1_indicators = ['{', '}', '::=', 'value', 'message', 'criticalExtensions',
                          'dl-CarrierFreq', 'q-RxLevMin', 't-ReselectionEUTRA',
                          'threshX-', 'allowedMeasBandwidth', 'cellReselectionPriority',
                          'sf-Medium', 'sf-High', 'presenceAntennaPort', 'neighCellConfig']
        
        return any(indicator in line for indicator in asn1_indicators)
    
    def _is_pure_hex_line(self, line: str) -> bool:
        """检查是否为纯HEX行"""
        # 移除空格和常见分隔符
        cleaned = re.sub(r'[\s\|\-\.]', '', line)
        if len(cleaned) < 8:  # 太短不算纯HEX
            return False
        
        # 检查是否全为HEX字符
        return all(c in "0123456789ABCDEFabcdef" for c in cleaned)
    
    def _is_hex_token(self, token: str) -> bool:
        """检查token是否为长HEX串"""
        if len(token) <= 4:
            return False
        
        # 移除0x前缀
        hex_part = token[2:] if token.startswith('0x') else token
        
        # 检查是否为长HEX串（>4字符且全为HEX）
        return (len(hex_part) > 4 and 
                all(c in "0123456789ABCDEFabcdef" for c in hex_part))
    
    def clean_sentence(self, sent) -> Tuple[str, bool]:
        """
        清洗单个句子
        
        Args:
            sent: spaCy Span对象
            
        Returns:
            (清洗后的句子文本, 是否保留)
        """
        # 过滤掉长HEX token
        tokens = []
        for token in sent:
            if not self._is_hex_token(token.text):
                tokens.append(token.text)
        
        if len(tokens) < 3:  # 太短
            return "", False
        
        sent_text = " ".join(tokens)
        
        # 长度检查 - 增加限制以保留ASN.1结构
        if len(sent_text) > 2000:  # 从300增加到2000字符
            return "", False
        
        # 检查是否为有意义的技术句子
        if self._has_technical_content(sent_text):
            return sent_text.strip(), True
        
        return "", False
    
    def _has_technical_content(self, text: str) -> bool:
        """检查文本是否包含技术内容"""
        text_lower = text.lower()
        
        # 包含技术关键词
        technical_keywords = [
            'rrc', 'mac', 'phy', 'pdcp', 'rlc', 'pdu', 'message',
            'procedure', 'event', 'value', 'asn.1', 'decoded',
            'uarfcn', 'psc', 'cell', 'frequency', 'power', 'sib', 'mib',
            'bcch', 'dcch', 'ccch', 'carrierfreq', 'reselection',
            'threshold', 'bandwidth', 'priority', 'antenna', 'config'
        ]
        
        if any(keyword in text_lower for keyword in technical_keywords):
            return True
        
        # ASN.1结构特征
        if any(pattern in text for pattern in ['::=', '{', '}', 'dl-', 'ul-', 'sf-']):
            return True
        
        # 包含数值和单位
        if re.search(r'\d+\s*(?:db|hz|khz|mhz|ghz|ms|sec)', text_lower):
            return True
        
        # 包含协议字段
        if re.search(r'[a-z]+_[a-z]+|[A-Z]{2,}', text):
            return True
        
        return False
    
    def _extract_clean_lines(self, rec: List[str]) -> List[str]:
        """
        从原始记录中提取干净、有意义的句子/块。
        1. 将整个ASN.1块提取为单个字符串。
        2. 将所有其他描述性行收集到另一个单独的字符串中。
        3. 过滤掉所有HEX dump行。
        """
        descriptive_lines = []
        asn1_blocks = []
        i = 0
        
        # 用于查找任何纯HEX、空格和常见前缀的行的正则表达式
        hex_line_re = re.compile(r"^\s*((Length:|Header:|Payload:)\s*)?([0-9A-Fa-f]{2}\s*)+\s*$")
        hex_prefix_re = re.compile(r"^\s*(Length:|Header:|Payload:|HEX Dump)")

        while i < len(rec):
            line = rec[i]

            # 跳过任何明显是HEX dump或其头部的行
            if hex_prefix_re.match(line) or hex_line_re.match(line):
                i += 1
                continue

            # 遇到Interpreted PDU块
            if line.strip().startswith("Interpreted PDU:"):
                j = i + 1
                # 寻找'value'块的开始
                while j < len(rec) and not rec[j].lstrip().startswith("value"):
                    # 我们可以将"Interpreted PDU"和"value"之间的行添加到描述性行中
                    if rec[j].strip():
                        descriptive_lines.append(rec[j])
                    j += 1
                
                if j < len(rec):
                    # 找到'value'，现在提取整个块
                    brace_depth = 0
                    rrc_block_lines = []
                    k = j
                    while k < len(rec):
                        r_line = rec[k]
                        brace_depth += r_line.count("{") - r_line.count("}")
                        rrc_block_lines.append(r_line.strip())
                        if brace_depth == 0 and '}' in r_line:
                            # 确保在括号匹配的行结束
                            if r_line.strip().endswith("}"):
                                break
                        k += 1
                    
                    asn1_blocks.append(" ".join(rrc_block_lines))
                    i = k + 1
                else:
                    # 没有找到'value'，继续
                    i = j
                continue

            # 这是一个普通的描述性行
            if line.strip():
                descriptive_lines.append(line)
            i += 1

        # 合并结果
        final_cleaned_lines = []
        if descriptive_lines:
            final_cleaned_lines.append(" ".join(descriptive_lines))
        final_cleaned_lines.extend(asn1_blocks)
        
        return final_cleaned_lines

    def _split_sentences(self, lines: List[str]) -> List[str]:
        """
        接收清理过的行/块，并通过spaCy的sentencizer运行它们。
        ASN.1块按原样传递。
        """
        sents = []
        for ln in lines:
            # 如果是预编译的ASN.1块，则不再进一步拆分。
            if ln.lstrip().startswith("value") and "::=" in ln:
                sents.append(ln)
            else:
                # 这是描述性块，让spaCy在需要时进行拆分。
                doc = self.nlp(ln)
                sents.extend(sent.text.strip() for sent in doc.sents if sent.text.strip())
        return sents

    def process_file(self, file_path: Path) -> List[str]:
        """
        处理单个日志文件
        
        Args:
            file_path: 日志文件路径
            
        Returns:
            提取的句子列表
        """
        try:
            # 读取文件
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
            # 按记录处理
            for record in self.iterate_records(raw_text):
                clean_lines = self._extract_clean_lines(record)
                sentences = self._split_sentences(clean_lines)
                all_sentences.extend(sentences)
            
            return all_sentences
            
        except Exception as e:
            print(f"[warn] Error reading {file_path}: {e}")
            return []
    
    def get_stats(self) -> dict:
        """获取处理器统计信息"""
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
    """测试处理器功能"""
    # 创建测试数据
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
    
    # 测试记录分割
    records = list(processor.iterate_records(test_log))
    print(f"Found {len(records)} records")
    
    # 测试句子提取
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
    
    # 统计信息
    stats = processor.get_stats()
    print(f"\nProcessor stats: {stats}")


if __name__ == "__main__":
    test_processor()
