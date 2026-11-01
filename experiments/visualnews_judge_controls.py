"""Caption interventions on manually inspected pixels; images never changed."""
import json
from visualnews_judge_pilot import OUT,save

CASES = [
    (199,'A woman in a grey jacket holds a microphone.','grey','bright red'),
    (744,'A football player in a red jersey numbered 58 holds up a cap.','red','blue'),
    (556,'Orange flames are visible in the windows of a building at night.','Orange','Green'),
    (707,'A group of people stand behind a green railing.','green','bright orange'),
    (545,'A small blue and white aircraft flies above fields.','blue and white','bright red'),
    (258,'A woman wearing sunglasses and a black shirt stands outdoors.','black','bright yellow'),
    (254,'A woman in a grey dress stands near photographers.','grey','bright red'),
    (703,'A man wearing glasses and a dark suit gestures with his hands.','dark','bright yellow'),
]

def main():
    sample={r['id']:r for r in json.loads((OUT/'sample.json').read_text(encoding='utf-8'))}
    controls=[]
    for num,base,color,wrong in CASES:
        source=sample[f'VisualNews_t2i:{num}']; base_id=f'control:{num}:base'
        def add(kind,caption,reference=base_id):
            controls.append(dict(source,id=f'control:{num}:{kind}',caption=caption,base_id=reference,control_kind=kind,
                intervention_only=True,inspection='Codex visual review of contact_0.jpg; visible color checked before model controls'))
        add('base',base)
        add('name',base[:-1]+', photographed by Evelyn Hart.')
        add('remove_name',base,f'control:{num}:name')
        add('date',base[:-1]+' on March 17, 2014.')
        add('event',base[:-1]+' during the opening of the 2014 Global Media Forum.')
        add('precise_location',base[:-1]+' at 1479 West Ash Street, Springfield, Illinois.')
        add('later_outcome',base+' Two days later, this photograph won the National Press Award.')
        add('wrong_color',base.replace(color,wrong))
    save(OUT/'controls.json',controls)
    save(OUT/'controls_protocol.json',dict(n_images=len(CASES),n_captions=len(controls),
        primary_checks=['external context strictly increases for date, event, precise location, later outcome versus clean visual base',
            'removing an unverifiable named photographer does not lower visual grounding',
            'wrong visible color creates mismatch when clean base has none'],
        caveat='Controls are curated development diagnostics, not a blinded accuracy benchmark. Synthetic event/date/location claims are unsupported, not presumed visually false.'))
    print('Prepared',len(controls),'caption controls',flush=True)

if __name__=='__main__':main()
