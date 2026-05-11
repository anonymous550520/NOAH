import re

def parse_hallucination_text(text):
    """
    Parse the structured output from hallucination evaluation prompt.
    Expected format contains EXTRACTED_EVENTS, CRITERIA_REVIEW, REASONING, and DETECTION sections.
    """
    if not text or not isinstance(text, str):
        return {
            "extracted_events": [],
            "criteria_review": "",
            "reasoning": "",
            "hallucination_count": 0,
            "parse_error": "Empty or invalid input text"
        }
    
    try:
        # Extract EXTRACTED_EVENTS section (with or without asterisks)
        events_pattern = r'\*{0,2}EXTRACTED_EVENTS:\*{0,2}\s*\n(.*?)(?=\n\*{0,2}CRITERIA_REVIEW:)'
        events_match = re.search(events_pattern, text, re.DOTALL)
        extracted_events = []
        
        if events_match:
            events_text = events_match.group(1).strip()
            # Parse numbered events (1. event, 2. event, etc.)
            event_lines = re.findall(r'^\s*\d+\.\s+(.+)$', events_text, re.MULTILINE)
            extracted_events = [event.strip() for event in event_lines]
        
        criteria_pattern = r'\*{0,2}CRITERIA_REVIEW:\*{0,2}\s*\n(.*?)(?=\n\*{0,2}EVENT-BY-EVENT\s+REASONING:)'
        criteria_match = re.search(criteria_pattern, text, re.DOTALL)
        criteria_review = criteria_match.group(1).strip() if criteria_match else ""

        reasoning_pattern = r'\*{0,2}EVENT-BY-EVENT\s+REASONING:\*{0,2}\s*\n(.*?)(?=\n\*{0,2}FINAL\s+METRICS:)'
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

        detection_pattern = r'HALLUCINATION[_ ]COUNT[:：]?\s*(\d+)'
        detection_match = re.search(detection_pattern, text)
        hallucination_count = int(detection_match.group(1)) if detection_match else 0
        
        return {
            "extracted_events": extracted_events,
            "criteria_review": criteria_review,
            "reasoning": reasoning,
            "hallucination_count": hallucination_count,
            "parse_error": None
        }
        
    except Exception as e:
        return {
            "extracted_events": [],
            "criteria_review": "",
            "reasoning": "",
            "hallucination_count": 0,
            "parse_error": f"Parsing failed: {str(e)}"
        }

def parse_omission_text(text):
    """
    Parse the structured output from omission evaluation prompt.
    Expected format contains GROUND_TRUTH_EVENTS, CRITERIA_REVIEW, EVENT-BY-EVENT REASONING, and FINAL METRICS sections.
    """
    if not text or not isinstance(text, str):
        return {
            "ground_truth_events": [],
            "criteria_review": "",
            "reasoning": "",
            "total_omission_count": 0,
            "inserted_omission_count": 0,
            "parse_error": "Empty or invalid input text"
        }
    
    try:
        # Extract GROUND_TRUTH_EVENTS section
        events_pattern = r'\*{0,2}GROUND[_ ]TRUTH[_ ]EVENTS\*{0,2}[:：]?\s*\n(.*?)(?=\n\s*\*{0,2}CRITERIA[_ ]REVIEW\*{0,2}[:：]?|$)'
        events_match = re.search(events_pattern, text, re.DOTALL)
        ground_truth_events = []
        
        if events_match:
            events_text = events_match.group(1).strip()
            # Parse numbered events (1. event, 2. event, etc.)
            event_lines = re.findall(r'^\s*\d+\.\s+(.+)$', events_text, re.MULTILINE)
            ground_truth_events = [event.strip() for event in event_lines]
        
        # Extract CRITERIA_REVIEW section
        criteria_pattern = r'\*{0,2}CRITERIA[_ ]REVIEW\*{0,2}[:：]?\s*\n(.*?)(?=\n\s*\*{0,2}EVENT[_-]*BY[_-]*EVENT[_ ]REASONING\*{0,2}[:：]?|$)'
        criteria_match = re.search(criteria_pattern, text, re.DOTALL)
        criteria_review = criteria_match.group(1).strip() if criteria_match else ""
        
        # Extract EVENT-BY-EVENT REASONING section
        reasoning_pattern = r'\*{0,2}EVENT[_-]*BY[_-]*EVENT[_ ]REASONING\*{0,2}[:：]?\s*\n(.*?)(?=\n\s*\*{0,2}FINAL[_ ]METRICS\*{0,2}[:：]?|$)'
        reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Extract TOTAL_OMISSION_COUNT and INSERTED_OMISSION_COUNT from FINAL METRICS section
        total_omission_pattern = r'TOTAL[_ ]OMISSION[_ ]COUNT[:：]?\s*(\d+)'
        total_omission_match = re.search(total_omission_pattern, text)
        total_omission_count = int(total_omission_match.group(1)) if total_omission_match else 0
        
        inserted_omission_pattern = r'INSERTED[_ ]OMISSION[_ ]COUNT[:：]?\s*(\d+)'
        inserted_omission_match = re.search(inserted_omission_pattern, text)
        inserted_omission_count = int(inserted_omission_match.group(1)) if inserted_omission_match else 0
        
        return {
            "ground_truth_events": ground_truth_events,
            "criteria_review": criteria_review,
            "reasoning": reasoning,
            "total_omission_count": total_omission_count,
            "inserted_omission_count": inserted_omission_count,
            "parse_error": None
        }
        
    except Exception as e:
        return {
            "ground_truth_events": [],
            "criteria_review": "",
            "reasoning": "",
            "total_omission_count": 0,
            "inserted_omission_count": 0,
            "parse_error": f"Parsing failed: {str(e)}"
        }