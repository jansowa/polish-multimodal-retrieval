"""Run with the existing WSL TinyCLIP environment; all writes use private temp dirs."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
from wiki_compact_archive import Archive,PROCESSOR_PATH,PROCESSOR_CONFIG_SHA256,digest,publish_absent,sha1_hex


class CompactArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import CLIPImageProcessor
        cls.processor=CLIPImageProcessor.from_pretrained(PROCESSOR_PATH,local_files_only=True)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='wiki-archive-test-');self.root=Path(self.temp.name)
        self.store=self.root/'store';self.store.mkdir()
        self.archive=Archive(self.store,processor=self.processor,external_catalogs=[])
        rng=np.random.default_rng(31)
        pixels=rng.integers(0,256,(613,1701,3),dtype=np.uint8)
        stream=io.BytesIO();Image.fromarray(pixels).save(stream,'JPEG',quality=96)
        self.raw=stream.getvalue();self.h=hashlib.sha1(self.raw).hexdigest()
        self.record=dict(id='wikimedia:test',source_sha1=self.h,image_original_url='https://upload.wikimedia.org/test.jpg',
            source_page_url='https://commons.wikimedia.org/wiki/File:Test.jpg',license='CC BY 4.0',
            license_url='https://creativecommons.org/licenses/by/4.0/',creator='Autor',attribution='Autor, CC BY 4.0',width=1701,height=613)

    def tearDown(self):self.archive.close();self.temp.cleanup()

    def create(self):
        result=self.archive.begin(self.record);self.assertEqual(result['action'],'download')
        return self.archive.commit(result['token'],self.raw)

    def test_original_fingerprints_and_lossless_native_from_original(self):
        result=self.create();receipt=result['receipt']
        self.assertEqual(receipt['sha256'],digest(self.raw))
        self.assertEqual(receipt['original_sha1'],self.h)
        self.assertEqual(receipt['original_dimensions'],[1701,613])
        self.assertEqual(receipt['master_dimensions'],[1536,554])
        self.assertTrue(receipt['preprocessing_bit_exact'])
        with Image.open(self.archive.paths(self.h)[1]) as native,Image.open(io.BytesIO(self.raw)) as original:
            expected=self.processor(images=original.convert('RGB'),return_tensors='np')['pixel_values']
            actual=self.processor(images=native,return_tensors='np')['pixel_values']
            np.testing.assert_array_equal(expected,actual)
        source=json.loads(Path(result['metadata_path']).read_text())
        self.assertEqual(source['attribution'],self.record['attribution'])

    def test_reference_profile_matches_every_image_fingerprint_and_encoded_derivative(self):
        sys.path.insert(0,str(ROOT/'experiments/data_study_20261030/wiki_recovery_20261031'))
        import image_assets as reference
        old_store=reference.STORE
        try:
            reference.STORE=self.root/'reference'
            # The reference encoder needs extra workspace on this deliberately
            # high-entropy fixture; this changes its buffer, not JPEG parameters.
            save=Image.Image.save
            def buffered_save(image,fp,format=None,**params):
                if format=='JPEG':
                    with patch('PIL.ImageFile.MAXBLOCK',64*1024**2):return save(image,fp,format,**params)
                return save(image,fp,format,**params)
            with patch.object(Image.Image,'save',buffered_save):expected=reference.create(self.raw,self.h)
        finally:reference.STORE=old_store
        actual=self.create()['receipt']
        for key in ['original_sha1','sha256','pixels_sha256','phash','original_dimensions','master_sha256',
                    'master_dimensions','native_sha256','native_dimensions','processor_config_sha256']:
            self.assertEqual(actual[key],expected[key],key)

    def test_ready_assets_reused_without_processor_or_file_replacement(self):
        self.create();paths=self.archive.paths(self.h)
        before={p:(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
        with patch.object(self.archive,'build',side_effect=AssertionError('Do not regenerate valid assets')):
            self.assertEqual(self.archive.begin(self.record)['action'],'ready')
        self.assertEqual(before,{p:(p.read_bytes(),p.stat().st_mtime_ns) for p in paths})

    def test_sha1_mismatch_creates_no_receipt_or_images_and_releases_lease(self):
        token=self.archive.begin(self.record)['token']
        with self.assertRaisesRegex(ValueError,'SHA1'):self.archive.commit(token,b'wrong original')
        self.assertFalse(any(p.exists() for p in self.archive.paths(self.h)))
        self.assertEqual(self.archive.begin(self.record)['action'],'download')

    def test_corrupt_existing_artifact_is_preserved_and_never_redownloaded(self):
        self.create();master,_,receipt=self.archive.paths(self.h)
        master.write_bytes(b'corrupt test fixture')
        before=receipt.read_bytes()
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.archive.begin(self.record)
        self.assertEqual(master.read_bytes(),b'corrupt test fixture');self.assertEqual(receipt.read_bytes(),before)

    def test_external_catalog_owner_is_respected_even_without_sha1_lock(self):
        self.archive.external.add(self.h)
        self.assertEqual(self.archive.begin(self.record)['action'],'deferred')
        self.assertFalse(any(p.exists() for p in self.archive.paths(self.h)))

    def test_external_failed_receipt_is_preserved_and_reported(self):
        path=self.archive.paths(self.h)[2];path.parent.mkdir(parents=True)
        payload=json.dumps(dict(status='failed',sha1=self.h,reason='original unavailable')).encode()
        path.write_bytes(payload);self.archive.external.add(self.h)
        with self.assertRaisesRegex(ValueError,'failure receipt'):self.archive.begin(self.record)
        self.assertEqual(path.read_bytes(),payload)

    def test_other_process_sha1_lock_prevents_duplicate_download(self):
        program="""import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from wiki_compact_archive import Archive
a=Archive(Path(sys.argv[2]),external_catalogs=[]);print(json.dumps(a.begin(json.loads(sys.argv[3]))),flush=True);input();a.close()
"""
        child=subprocess.Popen([sys.executable,'-c',program,str(ROOT/'experiments'),str(self.store),json.dumps(self.record)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(json.loads(child.stdout.readline())['action'],'download')
            self.assertEqual(self.archive.begin(self.record)['action'],'deferred')
        finally:child.communicate('\n',timeout=20)
        self.assertEqual(child.returncode,0)

    def test_atomic_write_never_overwrites_existing_bytes(self):
        path=self.store/'immutable.bin';publish_absent(path,b'first');publish_absent(path,b'first')
        with self.assertRaises(ValueError):publish_absent(path,b'second')
        self.assertEqual(path.read_bytes(),b'first')

    def test_partial_commit_can_resume_without_replacing_correct_master(self):
        original=publish_absent
        def fail_native(path,payload):
            if str(path).endswith('.png'):raise OSError('simulated interrupted write')
            return original(path,payload)
        token=self.archive.begin(self.record)['token']
        with patch('wiki_compact_archive.publish_absent',side_effect=fail_native):
            with self.assertRaises(OSError):self.archive.commit(token,self.raw)
        master,native,receipt=self.archive.paths(self.h);stamp=master.stat().st_mtime_ns
        self.assertFalse(native.exists());self.assertFalse(receipt.exists())
        self.create();self.assertEqual(master.stat().st_mtime_ns,stamp)

    def test_small_original_is_not_upscaled_and_exif_policy_matches_reference(self):
        array=np.zeros((90,40,3),dtype=np.uint8);array[:45]=[230,40,20]
        image=Image.fromarray(array);exif=image.getexif();exif[274]=6
        stream=io.BytesIO();image.save(stream,'JPEG',exif=exif);raw=stream.getvalue();h=hashlib.sha1(raw).hexdigest()
        receipt,_,_=self.archive.build(raw,h)
        self.assertEqual(receipt['master_dimensions'],[90,40]);self.assertEqual(receipt['original_dimensions'],[40,90])
        self.assertTrue(receipt['preprocessing_bit_exact'])

    def test_sha1_normalization_and_path_safety(self):
        self.assertEqual(sha1_hex(self.h),self.h)
        n=int(self.h,16);digits='0123456789abcdefghijklmnopqrstuvwxyz';base36=''
        while n:base36=digits[n%36]+base36;n//=36
        self.assertEqual(sha1_hex(base36),self.h)
        with self.assertRaises(ValueError):sha1_hex('../outside')


if __name__=='__main__':unittest.main()
