"""Last bounded iteration: evidence thresholds for natural mismatch false alarms."""
from visualnews_judge_pilot import OUT,save

RULES = '''
Final contradiction audit (apply before image_caption_mismatch):
- An unverified claim and an invisible/occluded object are NOT positive contradictory evidence. "I cannot see it" is insufficient. Set mismatch=false unless a specific material visual property is incompatible with the caption. Wrong visible color, incompatible object/action, or a clearly wrong fully visible spatial relation can establish contradiction.
- Inspect reflections, glass planes, flat cardboard cutouts, screens, drawings and courtroom sketches before inferring real 3D people, duplicate objects or another setting. A sketch can depict a courtroom; photographic style is not required. Do not assume a blank wall behind a screen: the screen may hide a window.
- A diplomatic meeting's participants or clothing do not establish its country. A national committee can meet in local council chambers; a place-name label does not by itself contradict an institutional role. A named church can be located in a different city from the city in its name. Parse the caption grammar before asserting a geographic conflict.
- Visible foreign-language text can support an English translation. Different languages are not contradiction. Do not invent a translation or reading when letters are uncertain. Mark the unreadable assertion unsupported instead of claiming a banner says something else.
- A caption such as "Name Honor" does not necessarily claim the pictured person IS that name. Ambiguous fragments, source credits and event labels need contextual interpretation; do not create an identity contradiction the caption never asserted.
- Nearly black, glitched or visually unreadable images support little or no caption information. State the unusable visual evidence in evidence, lower confidence appropriately, and keep mismatch=false unless the usable part supplies an actual incompatible attribute. Missing visual evidence is not proof of a wrong pair.
- Preserve sensitivity to definite visual contradictions. For example, a clearly blue car contradicts a caption's red car; uncertainty about a date does not.
Do this audit for every example. Never default all captions to matching or all names to ungrounded. Preserve the other scoring rules and JSON schema.
'''

def main():
    dest=OUT/'B_taxonomy_v3.txt'
    if dest.exists(): raise RuntimeError('v3 is immutable once registered')
    dest.write_text((OUT/'B_taxonomy_v2.txt').read_text(encoding='utf-8')+'\n'+RULES,encoding='utf-8')
    judgments={
        282:('false','Visible flat cutout borders and scene support cardboard figures; skin texture is not proof of live people.'),
        323:('false','San Francisco modifies church name; caption places the church in Lima. No geographic contradiction.'),
        845:('false','Greek banner is compatible with translated English; language difference and inferred parade context do not prove mismatch.'),
        253:('false','Diplomatic clothing/meeting does not disprove London or Tunisia.'),
        400:('false','Central window frame and symmetric steeple scene are compatible with reflection; no bridge contradiction.'),
        266:('false','Courtroom sketch is compatible with people sitting in court. Absence of judge furniture is not contradiction.'),
        202:('false','Projection screen can hide a window; behind-screen wall/window not visible.'),
        987:('false','No clearly visible chicken does not prove chicken absent; many occluded/remote regions.'),
        237:('false','Local chamber sign does not disprove a congressional field hearing; institutional role not a geographic restriction.'),
        511:('uncertain','No obvious bridge; cannot confidently establish support or contradiction from this framing.'),
        660:('uncertain','OCR identifies Leo Thorsness but short caption Jeb Bush Honor need not assert pictured identity.'),
        593:('uncertain','Flag cloth could be partly occluded; not enough evidence to assert conflict confidently.'),
        861:('unusable','Decoded image is visual glitch; asset-quality failure should not imply semantic contradiction.'),
        538:('unusable','Decoded image almost entirely black; insufficient usable visual evidence.'),
    }
    save(OUT/'manual_mismatch_reviews.json',[dict(id=f'VisualNews_t2i:{k}',expected_mismatch=False if v[0]=='false' else None,status=v[0],note=v[1],reviewer='Codex',inspection='mismatches_v2.jpg and captions, before v3 outputs') for k,v in judgments.items()])
    save(OUT/'iteration_v3.json',dict(reason='Manual inspection of all 14 natural v2 mismatch candidates exposed unsupported inferences, reflection/cutout errors and linguistic/geographic misreadings.',
        changes=['explicit positive contradiction threshold','reflection/cutout/sketch/occlusion audit','no language or geographic inference as contradiction','unusable assets yield insufficient evidence rather than semantic mismatch'],
        evaluation='Same 360 pairs, 64 controls and three draws on fixed 100-record subset; nine adjudicable natural false-alarm cases plus five ambiguous/unusable cases. These are tuning diagnostics, not independent test accuracy.',
        strategy='Refine B, the v2 strategy with lower distance to the 60 manual grounding intervals and no severe interval errors. Stop after this v3 iteration and report remaining errors.'))

if __name__=='__main__':main()
