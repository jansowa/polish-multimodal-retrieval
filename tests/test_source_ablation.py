import unittest
import torch
from pl_mmb_training.source_ablation import no_coco_indices,Sampler,variants,STEPS,WINDOW


def row(uid,dataset='aro_visual_attribution',parents=None,pixels=None):
    return dict(id=uid,dataset=dataset,parent_ids=parents or ['vg:'+uid],
        lineage_verified=True,sha256='file-'+uid,target_id=pixels or 'pixels-'+uid,
        text_pl='caption-'+uid)


class SourceAblationTests(unittest.TestCase):
    def test_official_flickr_alias_excludes_vg_image_even_if_own_coco_id_is_null(self):
        metadata=[dict(image_id=1,flickr_id=10,coco_id=None),dict(image_id=2,flickr_id=10,coco_id=99),
            dict(image_id=3,flickr_id=30,coco_id=None)]
        kept,reasons=no_coco_indices([row('1'),row('3')],metadata,[])
        self.assertEqual(kept,[1])
        self.assertEqual(reasons['transitive_COCO_Flickr_pixel_or_file_alias'],1)

    def test_pixel_alias_excludes_indirect_coco_and_unknown_vg_cannot_enter(self):
        rows=[row('1',pixels='same'),row('c','mbeir_mscoco',['coco:99'],pixels='same'),row('missing')]
        kept,reasons=no_coco_indices(rows,[dict(image_id=1,coco_id=None)],[])
        self.assertEqual(kept,[])
        self.assertEqual(reasons['missing_official_VG_identity'],1)

    def test_external_news_known_coco_alias_is_a_hard_failure(self):
        coco=row('c','mbeir_mscoco',['coco:99'],pixels='same')
        news=row('n','visualnews',['visualnews:n'],pixels='same')
        with self.assertRaisesRegex(ValueError,'COCO alias'):
            no_coco_indices([coco],[],[],[news])

    def test_leave_one_out_preserves_all_retained_original_slots(self):
        news=[row(str(i),'visualnews') for i in range(13)]
        full=Sampler(dict(mode='replace',base_pool=list(range(8))),8,news,8)
        drop=Sampler(dict(mode='replace',base_pool=[0,2,4,6]),8,news,8)
        for _ in range(20):
            a,b=full.draw(10),drop.draw(10)
            for x,y in zip(a[:5].tolist(),b[:5].tolist()):
                if x%2==0:self.assertEqual(x,y)
                else:self.assertGreaterEqual(y,8)

    def test_sampler_restores_exactly_without_drawing_excluded_rows(self):
        news=[row(str(i),'visualnews') for i in range(13)]
        plan=dict(mode='half_add',base_pool=[2,4,6])
        a=Sampler(plan,8,news,8);a.draw(11)
        b=Sampler(plan,8,news,8);b.restore(a.state())
        actual=a.draw(257)
        torch.testing.assert_close(actual,b.draw(257))
        self.assertTrue(all(i in (2,4,6) or i>=8 for i in actual.tolist()))

    def test_fixed_budget_and_fourteen_registered_source_conditions(self):
        plans=variants(dict(A=[0],R=[1],S=[2],C=[3]),[0,1],4)
        self.assertEqual(len(plans),14)
        self.assertEqual(STEPS//WINDOW*((WINDOW-1)*256+145),1405780)
        self.assertEqual(plans['VN_nonCOCO_AR']['base_pool'],[0,1])


if __name__=='__main__':unittest.main()
