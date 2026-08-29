# Copyright (c) 2026. Part of the artifact for “Not on the Same Page:
# Uncovering Specification Inconsistencies in O-RAN Standards,” submitted to
# USENIX Security 2027. Restricted evaluation material. See NOTICE.

import argparse
import glob
import os
import re
import fitz  # pip install PyMuPDF
import unicodedata
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent
RAW_DATA = PIPELINE_ROOT / "data/raw"
PROCESSED_DATA = PIPELINE_ROOT / "data/processed"

PRETRAINING_PATHS = {
    "ORAN": RAW_DATA / "ORAN",
    "5G": RAW_DATA / "5G",
    "4G": RAW_DATA / "4G",
}

FINETUNING_PATHS = {
    "ORAN": RAW_DATA / "ORAN",
    "5G": RAW_DATA / "5G",
    "4G": RAW_DATA / "4G",
}


import json

class Preprocessor:  # PARAGRAPH MERGING FROM BLOCKS + FINETUNING FILTERS
    def __init__(self, output_path, net_type):
        self.output_path = output_path
        self.net_type = net_type
        self.file_count = 0
        self.title_font_threshold = 11
        self.min_font_threshold = 8
        self.min_block_width=99
        #self.title_regex = r"^(\d+(?:\.\d+)*)(\.\d+[a-zA-Z]?)? "
        self.title_regex = r"^(\d+(?:\.\d+)*)(\.\d+[a-zA-Z]?)?[\s:\-\.]+"
        self.section_index = []  # Full index of all section headings
        self.global_index = 0
        self.debug_out = f"{self.output_path}/out-debug.txt"

    def processAll(self, input_path, save_flat=False):

        if save_flat:
            flat_file = os.path.join(self.output_path, f'corpus_{self.net_type}.jsonl')
            open(flat_file, 'w').close()  # Clear file before appending
        
        all_data = {}
        #output_file = os.path.join(self.output_path, f'corpus_{self.net_type}.jsonl')
        #with open(output_file, 'w') as out_file:
        with open(self.debug_out, "w") as debug_file:
            for pdf in glob.glob(f"{input_path}/**/*.pdf", recursive=True):
                self.file_count += 1
                print(f"[{self.file_count}] {os.path.basename(pdf)}")
                pdf_data = self.processIt(pdf, debug_file, save_flat)
                all_data.update(pdf_data)
    
                # print(os.path.join(self.output_path, f'section_index_{self.net_type}.json'))
                # Optional: export section hierarchy
                # section_tree = self.build_section_tree()
                # with open(os.path.join(self.output_path, f'section_index_{self.net_type}.json'), 'w') as f:
                #    json.dump(section_tree, f, indent=2)

        output_file = os.path.join(self.output_path, f'corpus_{self.net_type}_hierarchical.json')
        with open(output_file, 'w') as out_file:
            json.dump(all_data, out_file, indent=2, ensure_ascii=True)
    
    def processIt(self, file_path, debug_file, save_flat=False):
        # block_count = 0
        self.extract_titles(file_path)
        pdf_basename = os.path.basename(file_path)
        
        #section_map = {} # section key -> section_object
        #unknown_section_counter = 0

        # Build tree and index for attaching paragraphs
        # Returns (tree_root_list, lookup_table_by_section_number)
        section_tree, section_lookup = self.build_section_tree(debug_file, include_paragraphs=True)

        unknown_section = {
            "section_number": None,
            "section_title": "Unsectioned",
            "paragraphs": [],
            "children": []
        }

        inside_uml = False
        inside_asn1 = False
        
        with fitz.open(file_path) as doc:
            for page_number, page in enumerate(doc, start=1):
                blocks = page.get_text("dict")["blocks"]
                paragraph_buffer = []
                paragraph_y_buffer = []
                for block_num, block in enumerate(blocks, start=1):
                    if "lines" not in block:
                        continue

                    block_width = block["bbox"][2] - block["bbox"][0]
                    spans = []
                    for line_num, line in enumerate(block["lines"], start=1):
                        fonts = [span["font"] for span in line["spans"]]

                        if "CourierNewPSMT" in fonts: # code or config
                            debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] detected font CourierNewPSMT\n")
                            continue
                        
                        sizes = [span["size"] for span in line["spans"]]
                        if not sizes:
                            continue
                        #print(fonts)
                        max_size = max(sizes)
                        text = "".join(span["text"] for span in line["spans"]).strip()
                        if text:

                            # enter/exit ASN.1 region
                            if "-- asn1start" in text.lower():
                                inside_asn1 = True
                                #if debug:
                                #print(fonts)
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Entering ASN.1 block: {text}\n")
                                continue
                            if "-- asn1stop" in text.lower():
                                inside_asn1 = False
                                #if debug:
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Exiting ASN.1 block: {text}\n")
                                continue
                            if inside_asn1:
                                #if debug:
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Skipped ASN.1 content: {text}\n")
                                continue
                            
                            if "@startuml" in text.lower():
                                inside_uml = True
                                #if debug:
                                #print(fonts)
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Entering UML block: {text}\n")
                                continue
                            if "@enduml" in text.lower():
                                inside_uml = False
                                #if debug:
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Exiting UML block: {text}\n")
                                continue
                            if inside_uml:
                                #if debug:
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Skipped UML content: {text}\n")
                                continue

                            if block_width < self.min_block_width:
                                #if debug:
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Skipped narrow block (width {block_width:.2f}): {text}\n")
                                continue
                            if max_size <= self.min_font_threshold:
                                #if debug:
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Skipped small font (size {max_size}): {text}\n")
                                continue
                            if text.lower().startswith(("table", "figure", "fig.", "code")):
                                #if debug:
                                debug_file.write(f"[Page {page_number} Block {block_num} Line {line_num}] Skipped labeled line: {text}\n")
                                continue
                            
                            spans.append(text)
                    if not spans:
                        continue

                    block_text = " ".join(spans)
                    #block_text = unicodedata.normalize("NFKC", block_text)

                    if len(block_text.split()) < 4 : #skip the block
                        debug_file.write(f"Skipping 0: {block_text}\n")
                        continue
                    if self.find_sections(block_text):
                        #print("Skipping 1:", block_text)
                        debug_file.write(f"Skipping 1: {block_text}\n")
                        continue
                    if "editor's note" in block_text.lower():
                        #print("Skipping 2:", block_text)
                        debug_file.write(f"Skipping 2: {block_text}\n")
                        continue
                    if re.search(r'©\s*20\d{2}\s+by the O-RAN ALLIANCE e\.V\.', block_text):
                        #print("Skipping 3:", block_text)
                        debug_file.write(f"Skipping 3: {block_text}\n")
                        continue
                    if re.match(r'^[_\-\s]{10,}$', block_text):
                        #print("Skipping 4:", block_text)
                        debug_file.write(f"Skipping 4: {block_text}\n")
                        continue
                    if re.match(r'^\d{1,4}$', block_text):
                        #print("Skipping 5:", block_text)
                        debug_file.write(f"Skipping 5: {block_text}\n")
                        continue
                    if re.match(r'^(?:O-?RAN)[\.-][A-Za-z0-9&\.\-_\s]+$', block_text):
                        debug_file.write(f"Skipping 6: {block_text}\n")
                        #print("Skipping 6:", block_text)
                        continue
                    if self.is_probable_table(block):
                        debug_file.write(f"[Page {page_number} Block {block_num}] Skipped probable table block: {block_text}\n")
                        continue

                    #block_text = re.sub(r' +', ' ', block_text)
                    #block_text = re.sub(r'\ue000', '', block_text, re.UNICODE)

                    # We place regex here if we are checking the beginning or end of the block
                    block_text = re.sub(r'^(->|[\.·•\-])', '', block_text) #starting dot, interpunct, hyphen removal
                    block_text = re.sub(r'[\.:,;•\-]+$', '.', block_text) #one or multiple dot, colon, hyphen, semicolon at the end converted to dot
                    block_text = self.remove_control_chars(block_text, debug_file)
                    block_text = self.normalize_unicode(block_text)
                    
                    #u_matches = re.findall(r'[\u0000-\u001F\u007F-\u009F]', block_text)
                    #if u_matches:
                    #    print(u_matches)

                    if not block_text:
                        continue

                    block_y = block.get("bbox", [None, None, None, None])[1]
                    #print("Block: {}, Page: {}, Y-pos: {}".format(block_text, page_number, block_y))

                    # Buffering logic
                    if not paragraph_buffer:
                        paragraph_y_buffer = [block_y]
                    paragraph_buffer.append(block_text)

                    # Check if the current line ends a paragraph
                    if re.search(r'[.!?]("|”)?$', block_text.strip()):
                        merged_text = " ".join(paragraph_buffer)
                        merged_text = self.apply_filters_after_merging(merged_text, debug_file)
                        merged_text = self.normalize_unicode(merged_text)
                        paragraph_buffer = []

                        y_for_match = min(paragraph_y_buffer) if paragraph_y_buffer else block_y
                        closest = self.get_closest_section_title(page_number, y_for_match)
                        #closest["number"] is the section number
                        section_node = section_lookup.get(closest["number"]) if closest else None

                        target = section_node if section_node else unknown_section
                        target["paragraphs"].append({
                            "text": merged_text,
                            "page_number": page_number
                        })
                        paragraph_y_buffer = []
                        
                # Final flush in case last paragraph didn’t end with punctuation
                if paragraph_buffer:
                    merged_text = " ".join(paragraph_buffer)
                    merged_text = self.apply_filters_after_merging(merged_text, debug_file)
                    if merged_text:
                        y_for_match = min(paragraph_y_buffer) if paragraph_y_buffer else 0
                        closest = self.get_closest_section_title(page_number, y_for_match)
                        section_node = section_lookup.get(closest["number"]) if closest else None
                        target = section_node if section_node else unknown_section
                        target["paragraphs"].append({
                            "text": merged_text,
                            "page_number": page_number
                        })

        full_tree = section_tree
        if unknown_section["paragraphs"]:
            full_tree.insert(0, unknown_section)

        # Optional: export flat .jsonl format
        if save_flat:
            flat_file = os.path.join(self.output_path, f'corpus_{self.net_type}.jsonl')
            with open(flat_file, 'a') as flat_out:
                for section in full_tree:
                    self.write_section_flat_with_ancestors(section, flat_out, pdf_basename)

        #print(f"   {self.file_count} files → {block_count} blocks added\n" + "-" * 80)
        return {pdf_basename: full_tree}

    def is_probable_table(self, block):
        # Count spans per line
        spans_per_line = [len(line["spans"]) for line in block.get("lines", []) if line.get("spans")]
        
        # Heuristic 1: average spans per line is high (e.g., >= 4)
        high_span_density = spans_per_line and (sum(spans_per_line) / len(spans_per_line) >= 4)
    
        # Heuristic 2: multiple lines start with numbers or bullets (typical for itemized data)
        bullet_lines = 0
        for line in block.get("lines", []):
            text = "".join(span["text"] for span in line.get("spans", [])).strip()
            if re.match(r'^(\d+[\.\)]|[-*•>]+)', text):
                bullet_lines += 1
        many_bullets = bullet_lines >= 2
    
        # Heuristic 3: mostly short lines (table cells), rarely full sentences
        all_texts = ["".join(span["text"] for span in line.get("spans", [])).strip() for line in block.get("lines", [])]
        short_lines = sum(len(text.split()) <= 6 for text in all_texts)
        high_short_ratio = short_lines / len(all_texts) > 0.7 if all_texts else False
    
        return high_span_density or many_bullets or high_short_ratio

    def apply_filters_after_merging(self, merged_text, debug_file):

        merged_text = re.sub(r'_{10,}', ' ', merged_text)
        # Remove clause references like "clause 5.2.2.1.1.3.2"
        #merged_text = re.sub(r'\b[Cc]lause\s+\d+(?:\.\d+)+', '', merged_text)
        #merged_text = re.sub(r'\b[Cc]lause\s+\d+(?:\.\d+)*', '', merged_text)
        merged_text = re.sub(
            r'\b(?:[Ss]ub[\-\s]?[Cc]lause|[Cc]lause)\s+\d+(?:\.\d+)*',
            '',
            merged_text
        )
        # Remove ASSET references like ASSET-C-33
        merged_text = re.sub(r'\bASSET-[A-Z]-\d+\b', '', merged_text)
    
        # Remove capitalized spec tags like SEC-CTL-OCLOUD-COT-3:
        merged_text = re.sub(r'\b[A-Z]{2,}(?:-[A-Z0-9]+){1,5}[:-]?', '', merged_text)
    
        # Remove [ASSET-*], [REQ-*], etc.
        merged_text = re.sub(r'\[\s*[A-Z]{2,}(?:-[A-Z0-9]+)+\s*\]', '', merged_text)
    
        # Remove references like O-RAN.WG3.XYZ [2], 3GPP TR 21.905 [i.1]
        merged_text = re.sub(r'\b(?:O-?RAN|3GPP)[\w\.\- ]*\[\s*[^\]]+\]', '', merged_text)

        merged_text = self.remove_oran_copyright(merged_text, debug_file)

        # Remove [REQ-123] or similar bracketed tags
        merged_text = re.sub(r'\[REQ[\s\-]*[A-Za-z0-9]+\]', '', merged_text, flags=re.IGNORECASE)

        # Remove REQ-SEC-SMO-1: or REQ-XXX-YYY-Z: patterns ending with colon
        merged_text = re.sub(r'\bREQ(?:-[A-Za-z0-9]+)+:', '', merged_text)
        
        merged_text = re.sub(r'\( ', '(', merged_text)  # whitespace after opening paren
        merged_text = re.sub(r' \)', ')', merged_text)  # whitespace before closing paren
        merged_text = re.sub(r'(\(\))|(\[\])|(\{\})', '', merged_text) #remove empty paren, curly-braces, brackets
        merged_text = re.sub(r'(as shown below|[Ss]ee figure below):*\-*', '', merged_text) #remove certain strings
        merged_text = re.sub(r'([,;:])(\w)', r'\1 \2', merged_text) #insert whitespace after punctuations, except fullstop, underscore
                                                      #hyphen
        merged_text = re.sub(r'\ue000', '', merged_text, re.UNICODE)

        # Remove fake section numbers like "5.2.3.1.4.3", "4 5.2.3.1.4.3.1", etc.
        merged_text = re.sub(
            r'\b\d{1,2}(?:\.\d{1,2}){2,}(?:\.\d+[a-zA-Z])?\b(?:\s+[A-Z][a-zA-Z ]{1,30})?',
            '',
            merged_text
        )

        # Step 2 (optional): Normalize whitespace
        merged_text = re.sub(r'\s+', ' ', merged_text).strip()

        return merged_text

    def normalize_unicode(self, text):
        replacements = {
            "\u2013": "-", "\u2014": "-", "\u2212": "-",  # normalize minus/dash
            "\u2217": "*",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
    
        # Then drop everything that’s still non-ASCII
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        return text.strip()
        
    def remove_oran_copyright(self, block_text, debug_file):
        patterns = [
            # 1. Annex-based Adopter License Agreement notice (Copyright or © or both)
            re.compile(
                r'(_+\s*)?(Copyright\s*)?(©)?\s*20\d{2}\s*(?:by\s+)?(?:the\s+)?O-?RAN\s+Alliance(?:\s*e\.?v\.?)?\.?\s*'
                r'Your use is subject to the terms of the O-?RAN Adopter License Agreement in\s*(?:the\s+)?Annex\s+[A-Z]{1,4}(?:\.\s*|\s+)?',
                flags=re.IGNORECASE
            ),
    
            # 2. Full disclaimer & redistribution clause block
            re.compile(
                r'Copyright\s+20\d{2}\s+(?:the\s+)?O-?RAN\s+Alliance\.\s+THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS.*?'
                r'without specific prior written permission\.\"?;',
                flags=re.IGNORECASE | re.DOTALL
            ),
    
            # 3. Combined copyright + cover page notice
            re.compile(
                r'Copyright\s*(©)?\s*20\d{2}\s+O-?RAN\s+Alliance(?:\s*e\.?v\.?)?\.?\s*Your use is subject to copyright statement on the cover page.*',
                flags=re.IGNORECASE
            ),
    
            # 4. Simple copyright line (with or without leading underscores)
            re.compile(
                r'(_+\s*)?Copyright\s*(©)?\s*20\d{2}\s*(?:by\s+)?(?:the\s+)?O-?RAN\s+Alliance(?:\s*e\.?v\.?)?\.?',
                flags=re.IGNORECASE
            ),
    
            # 5. Standalone cover page copyright notice
            re.compile(
                r'Your use is subject to (the\s+)?copyright statement on the cover page.*',
                flags=re.IGNORECASE
            ),
    
            # 6. © only version with Annex notice
            re.compile(
                r'(_+\s*)?©\s*20\d{2}\s+O-?RAN\s+Alliance(?:\s*e\.?v\.?)?\.?\s*Your use is subject to the terms of the O-?RAN Adopter License Agreement in\s*(?:the\s+)?Annex\s+[A-Z]{1,4}(?:\.\s*|\s+)?',
                flags=re.IGNORECASE
            ),
    
            # 7. Standalone Annex-only License line (newly added!)
            re.compile(
                r'Your use is subject to the terms of the O-?RAN Adopter License Agreement in\s*(?:the\s+)?Annex\s+[A-Z]{1,4}(?:\.\s*|\s+)?',
                flags=re.IGNORECASE
            ),
        ]
    
        for pattern in patterns:
            matches = pattern.findall(block_text)
            if matches:
                #print("⚠️ Removed the following match(es):")
                debug_file.write("⚠️ Removed the following match(es):")
                for match in pattern.finditer(block_text):
                    #print(f"➡️ '{match.group().strip()}'\n")
                    debug_file.write(f"➡️ '{match.group().strip()}'\n")
                block_text = pattern.sub('', block_text)
    
        return block_text.strip()


        
    #def remove_control_chars(self, text):
    #    # Remove all control characters except newline (\n) and tab (\t)
    #    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    def remove_control_chars(self, text, debug_file):
        # Define the control character pattern (excluding \n and \t)
        control_char_pattern = r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
    
        # Find all control characters in the text
        control_chars = re.findall(control_char_pattern, text)
        if control_chars:
            unique = sorted(set(control_chars))
            hex_repr = [f"\\x{ord(c):02x}" for c in unique]
            #print(f"Removed control characters: {hex_repr}")
            debug_file.write(f"Removed control characters: {hex_repr}\n")

        # Remove them from the text
        return re.sub(control_char_pattern, '', text)
        
    def extract_titles(self, pdf_path):
        self.section_index.clear()
        doc = fitz.open(pdf_path)

        for page_number, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                block_title = []
                block_y = block.get("bbox", [None, None, None, None])[1]
                if "lines" in block:
                    sizes = [span["size"] for line in block["lines"] for span in line["spans"]]
                    max_size = max(sizes) if sizes else 0
                    if max_size > self.title_font_threshold:
                        for line in block["lines"]:
                            text = " ".join(span["text"].strip() for span in line["spans"]).strip()
                            if text:
                                block_title.append(text)

                if not block_title:
                    continue
                
                full_title_raw = " ".join(block_title)
                match = re.match(self.title_regex, full_title_raw)

                if not match:
                    continue
                
                section_number = match.group(1)
                section_level = self.get_section_level(section_number) if section_number else None

                if section_number:
                    # # SECTION_REGEX_HERE
                    # Remove "3.1 ", "2.4 -", "5: " etc.
                    cleaned_title = re.sub(rf"^{re.escape(section_number)}(\s+|:|-)\s*", "", full_title_raw)
                    #print(cleaned_title)
                    
                else:
                    cleaned_title = full_title_raw

                cleaned_title = self.normalize_unicode(cleaned_title)
                full_title_raw = self.normalize_unicode(full_title_raw)

                self.section_index.append({
                    "page": page_number,
                    "y": block_y,
                    "title": cleaned_title,     # cleaned version (no number)
                    "raw": full_title_raw,      # original full heading
                    "number": section_number,
                    "level": section_level
                })

        #print("Section Index:\n{}".format(self.section_index))

        #self.section_index.sort(key=lambda s: (s["page"], s["y"]))

    def build_section_tree(self, debug_file, include_paragraphs=False):
        root = []
        section_map = {}
        lookup_table = {}
    
        for section in self.section_index:
            node = {
                "section_number": section["number"],
                "section_title": section["title"],
                "page": section["page"],
                "y": section["y"],
                "level": section["level"],
                "children": []
            }
    
            if include_paragraphs:
                node["paragraphs"] = []
    
            section_number = section["number"]
            if section_number is None:
                root.append(node)
                continue
    
            if section_number in section_map: # - DEBUG
                #print(f"Duplicate section number: {section_number} on page {section['page']}")
                debug_file.write(f"Duplicate section number: {section_number} on page {section['page']}\n")
            section_map[section_number] = node
            lookup_table[section_number] = node
    
            if "." not in section_number:
                root.append(node)
            else:
                parent_number = ".".join(section_number.split(".")[:-1])
                parent_node = section_map.get(parent_number)
                if parent_node:
                    parent_node["children"].append(node)
                else:
                    root.append(node)
    
        return (root, lookup_table) if include_paragraphs else root
    
    def find_sections(self, line):
        # Match against original unmodified headings
        return any(line.strip() == s["raw"] for s in self.section_index)

    def get_closest_section_title(self, page_number, y_position):
        best_match = None
        best_page = -1
        best_y = -float("inf")

        for section in self.section_index:

            #print(section)
            
            s_page = section["page"]
            s_y = section["y"]

            if s_page > page_number:
                break
            if s_page == page_number and s_y >= y_position:
                break

            if s_page > best_page or (s_page == best_page and s_y > best_y):
                best_page = s_page
                best_y = s_y
                best_match = section

        #print("Section Found: {}".format(best_match))
        #print("----------------------------------------------------------------")
        return best_match

    def extract_section_number(self, title):
        #match = re.match(r"^(\d+(\.\d+)*)(\s+|:|-)", title or "") # SECTION_REGEX_HERE
        match = re.match(self.title_regex, title or "") # SECTION_REGEX_HERE

        #if match["match"] != match2["match"]:
        #    print(match, match2)
        
        #print("Section number match:", match.group(0))
        #print("Section number match:", match2)

        #if match is not None and match2 is not None:
        #    if match.group(0).strip() != match2.group(0).strip():
        #        print(match.group(0), match2.group(0))
        #elif match != match2: # one of them is None, not both
        #    print(match, match2)
        
        if match:
            return match.group(1)
        return None

    def get_section_level(self, section_number):
        if not section_number:
            return None
        return section_number.count('.') + 1

    def write_section_flat_with_ancestors(self, section, out_handle, pdf_basename, ancestors=None):
        if ancestors is None:
            ancestors = []
    
        current_ancestor_entry = {
            "section_number": section["section_number"],
            "section_title": section["section_title"],
            "level": section.get("level")
        } if section["section_number"] is not None else None
    
        updated_ancestors = ancestors + ([current_ancestor_entry] if current_ancestor_entry else [])
    
        for para in section.get("paragraphs", []):
            out_handle.write(json.dumps({
                "id": self.global_index,
                "pdf_file": pdf_basename,
                "section_number": section["section_number"],
                "section_title": section["section_title"],
                "section_level": section.get("level"),
                "ancestors": ancestors,
                **para
            }, ensure_ascii=True) + "\n")
            self.global_index += 1
    
        for child in section.get("children", []):
            self.write_section_flat_with_ancestors(child, out_handle, pdf_basename, updated_ancestors)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess specification PDFs for pretraining and/or finetuning."
    )

    parser.add_argument(
        "--net-type",
        choices=["ORAN", "5G", "4G"],
        default="ORAN",
        help="Network/specification type to process. Default: ORAN",
    )

    parser.add_argument(
        "--pretraining",
        action="store_true",
        help="Process the pretraining corpus.",
    )

    parser.add_argument(
        "--finetuning",
        action="store_true",
        help="Process the finetuning corpus.",
    )

    parser.add_argument(
        "--input-dir",
        help="Override the default input directory.",
    )

    parser.add_argument(
        "--output-dir",
        help="Override the default output directory.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    net_type = args.net_type

    run_pretraining = args.pretraining
    run_finetuning = args.finetuning

    if not run_pretraining and not run_finetuning:
        if net_type in PRETRAINING_PATHS:
            run_pretraining = True
            run_finetuning = True
        else:
            run_finetuning = True

    if args.input_dir and run_pretraining and run_finetuning:
        raise ValueError(
            "--input-dir cannot be used when running both pretraining and "
            "finetuning because they require separate input directories."
        )

    if args.output_dir and run_pretraining and run_finetuning:
        raise ValueError(
            "--output-dir cannot be used when running both pretraining and "
            "finetuning because they require separate output directories."
        )

    if run_pretraining:
        if net_type not in PRETRAINING_PATHS and not args.input_dir:
            raise ValueError(
                f"Pretraining is not configured for {net_type}. "
                f"Available types: {', '.join(PRETRAINING_PATHS)}"
            )

        input_path = args.input_dir or PRETRAINING_PATHS[net_type]
        output_path = (
            args.output_dir
            or PROCESSED_DATA / net_type
        )

        os.makedirs(output_path, exist_ok=True)

        pretrain_processor = Preprocessor(output_path, net_type)
        pretrain_processor.processAll(input_path, save_flat=True)

    if run_finetuning:
        input_path = args.input_dir or FINETUNING_PATHS[net_type]
        output_path = args.output_dir or PROCESSED_DATA / net_type

        os.makedirs(output_path, exist_ok=True)

        finetuning_processor = Preprocessor(output_path, net_type)
        finetuning_processor.processAll(input_path, save_flat=True)


if __name__ == "__main__":
    main()
