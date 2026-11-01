"""Register a substantive v2 after v1 diagnostic review; preserve every v1 prompt."""
import json
from visualnews_judge_pilot import OUT, COMMON, STRATEGIES, save

REFINEMENT = '''
Calibration rules from diagnostic review:
1. Explicitly credit matching visible actions, objects and attributes before penalizing context. A name or date does not erase a matching visual scene. Use 0 only when the caption communicates no supported material proposition; use 2 when a substantial concrete description coexists with substantial unsupported context. Do not automatically assign 1 merely because a name is present.
2. Source credits ("image courtesy of X", "image via X") describe provenance, not the depicted scene. They are mostly_contextual with grounding 0 unless a legible credit actually establishes that provenance. A graph or photo in the image does not make its attribution visually grounded.
3. Do not turn facial recognition, movie familiarity or newsroom memory into verified identity. Readable labels may establish the words they display; they do not prove every biographical claim. Distinctive unmistakable landmarks may support a place, but generic interiors and inferred cities do not. Visible occupation/activity is different from a named official title.
4. Visible text is valid evidence: quote the relevant short text in evidence if relying on it and set ocr_dependency=true. Evaluate maps, graphs, diagrams and screenshots on the same rules; never treat their captions as automatically unverifiable.
5. Before marking mismatch, identify an actual conflicting visual attribute in evidence. An unverified event, location, source credit, injury, motive or name alone is not contradiction. Partial occlusion or an ambiguous count is insufficient to assert mismatch confidently.
6. 4 requires no material unsupported information; evidence and unsupported_information must agree with the score. Distinguish a future plan/outcome from an action shown now. Specific event assertions remain contextual even when a related generic action is visible.
Return the populated JSON instance only. Do not repeat the template or output markdown.
'''

def main():
    common=COMMON.replace('Familiarity with a face, historical photo, news story or venue', 'Familiarity with a face, historical photo or news story')
    for base in ['B_taxonomy','C_claims']:
        path=OUT/(base+'_v2.txt')
        if path.exists(): raise RuntimeError('v2 already registered; prompts are immutable')
        path.write_text(common+'\n'+STRATEGIES[base]+'\n'+REFINEMENT,encoding='utf-8')
    save(OUT/'iteration_v2.json',dict(changes=['credit positive visual propositions before contextual penalty','source credits are provenance, not visual descriptions','explicit OCR and landmark evidence rules','mismatch requires positive conflicting evidence','native schema-constrained output'],
        reason='v1 overcredits identity/source provenance in several manually inspected cases and undercredits concrete scenes; raw JSON validity differs greatly across strategies.',
        design='Two revised strategies retain common scales and schema. Structured decoding is a separate execution change: do not attribute improved JSON validity solely to prompt wording.',
        evaluation='All variants use identical 360 natural pairs; 64 curated controls; three draws on the same 100 records. Exploratory diagnostics, not a blinded holdout accuracy claim.'))

if __name__=='__main__':main()
