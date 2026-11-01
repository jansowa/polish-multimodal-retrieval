import unittest
from guard import filter_verified


class GuardTests(unittest.TestCase):
    def row(self,uid,**kw):
        return dict(id=uid,sha256='raw-'+uid,pixels_sha256='pixel-'+uid,
                    phash='ffffffffffffffff',lineage_verified=True,**kw)

    def test_image_only_exact_alias(self):
        p=self.row('dev');r=self.row('alias');r['sha256']=p['sha256']
        kept,rejected=filter_verified([r],[p])
        self.assertFalse(kept);self.assertEqual(len(rejected),1)

    def test_transitive_text_then_image_only(self):
        p=self.row('dev',text_pl=' Chroniony  TEKST ')
        bridge=self.row('bridge',text_pl='chroniony tekst',parent_ids=['source:alias'])
        image_only=self.row('image',parent_ids=['source:alias'])
        bridge['phash']='0000000000000000';image_only['phash']='000000000000ffff'
        kept,rejected=filter_verified([image_only,bridge],[p])
        self.assertFalse(kept);self.assertEqual(len(rejected),2)

    def test_phash_boundary_and_clean(self):
        p=self.row('dev');near=self.row('near');far=self.row('far')
        near['phash']='ffffffffffffffc0' # 6 bits differ
        far['phash']='0000000000000000'
        kept,rejected=filter_verified([near,far],[p])
        self.assertEqual([r['id'] for r in kept],['far'])
        self.assertEqual([r['id'] for r,_ in rejected],['near'])

    def test_transitive_phash(self):
        p=self.row('dev');b=self.row('b');c=self.row('c')
        b['phash']='ffffffffffffffc0';c['phash']='fffffffffffff000' # 12 from dev, 6 from bridge
        kept,rejected=filter_verified([c,b],[p])
        self.assertFalse(kept);self.assertEqual(len(rejected),2)

    def test_unknown_fingerprint_rejected(self):
        r=self.row('unknown');r.pop('phash')
        self.assertFalse(filter_verified([r],[])[0])


if __name__=='__main__':unittest.main()
