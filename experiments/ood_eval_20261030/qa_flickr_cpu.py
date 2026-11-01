"""Apply preregistered surface/language QA also to repack-provided Flickr MT."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
from collections import Counter
from prepare import OUT, read, write, load_common, committed_bytes, sha
from acquire import ASSETS
from translate_cpu import qa


def main():
    common=load_common()
    decision=read(OUT/'translation_decision.json')
    write(OUT/'flickr_translation_QA_decision.json',dict(status='before_retrieval_scores',
        inherited_surface_rules=decision['qa_rules'],language_detector=decision['language_detector'],
        eligibility='Whole image ineligible if any original caption has unresolved QA flags; no hand editing or replacement.',
        ready_MT='NLLB-200-1.3B from repack; exact original translator revision unavailable'))
    sys.path.insert(0,str(ASSETS/'language_qa/vendor'))
    from langid.langid import LanguageIdentifier,model
    identifier=LanguageIdentifier.from_modelstring(model,norm_probs=True)
    identifier.set_languages(['pl','en','cs','sk','de','ru'])
    records=read(ASSETS/'flickr30k/candidate_images.json')
    result=[]
    for image in records:
        qualities=[dict(id=caption['id'],**qa(caption['text_en'],caption['text_pl'],identifier)) for caption in image['captions']]
        result.append(dict(id=image['id'],accepted=all(row['accepted'] for row in qualities),captions=qualities))
    common.reserve('B','flickr-QA',{'C':4*1024**2},committed_bytes=committed_bytes())
    path=ASSETS/'flickr30k/translation_qa.json'
    common.atomic_json(path,result)
    common.reserve('B','flickr-QA',{'C':0},committed_bytes=committed_bytes())
    summary=dict(status='automatic_QA_complete',images=len(result),accepted_images=sum(row['accepted'] for row in result),
        caption_flags=dict(Counter(flag for row in result for caption in row['captions'] for flag in caption['flags'])),
        artifact=common.artifact('Flickr_ready_MT_QA',path), decision_sha256=sha(OUT/'flickr_translation_QA_decision.json'))
    write(OUT/'flickr_translation_QA_summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
