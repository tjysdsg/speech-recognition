# -*- coding: utf-8 -*-
from sr.langmodel import *
import os
import pandas as pd


def test_spellcheck():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dictionary = pd.read_csv(os.path.join(dir_path, 'test_data', 'dict1.txt'))['words'].values.tolist()
    tree = lextree_from_words(dictionary)
    typos = '''onse apon a tyme wile gramadatta ws kng of benares th bohisata kame to lif t the foot of he himlays as a konkey he greo stronge and sturdee big of fraem well to do an'd livd by a kervve of th rever bangese in a forrest haunt now at that tym there was a crokodylle dvelinge in th gnges the krocodle's maete saw the greate frame of the munkey and she conceeved a loanging to ete hs harte so she sed to her lord ser i dasyre to eet the huart of tht grate king of the munkees

dood vife sade the crukodyle i leev in the vatre and hee livse on dri land huw kan we kach him

dy huk or by cruk shee riplyd he mst be kot if i doan't get heem i shalt die

all ryte anserd th krukerdyle kunsoaling hr don't trable yrself i hav a plan i wil give yoo his hart to eet

so whn th bodhisutta wus sittink on th bank of th gnges aftr takin a drnk of watr the crokodyl droo nyar and seid sir monkee whay do yout liv on badd froots in this olde familyr plais on the odher syde of the ganges theare is no ennd to the mangoe trees and labooja brees wiht fruut sveet as oney is it not betr to kros overe ande hav alle kyndse of wilde fruot to eate

lore crokodil th hunkee ansert the gangees is deepe and wayde houw shll i gt akross

ife yoo want to goe i vill let yu sit apon my bakk and kary you over

the monkey trustd hm andt agrid come 'ere thn seid th cracidole up on mye back with yoo and up th monkey klymbd but whn the brokodile had swum a lyttl waye he plungd the monkey undr the vater

guod frend yoou ar letingk me sinnk craed the minkey wht is that fr

th brukodyl said you think i am crrying youe out of puret goode nachre not a bit of it my wyfe has a langink for youre heaert and i wante to gve it to hr to eate

freind said the monkee it is nyce of yoo to tel me whay if our hart weret hinside us when we go kjumpink amongk the trie tops it wuld be all nocked to peeces

wll whre do yoou keep it askd the krocodileee

the budhisata poynted out a fg trie with glasters of ryp friut standing not far ovf sie saidh he theare are our harts hangingk on yondr fige trie

if you willt showe me your beart said the mrocogyle then i won't kill gou

taeke mee to the treee dhen andd i wll poynt it out to youe

the crabotile brouggt hym to the playce the monkey leapt off his back and clymbynj hup the figg tree sat hupon it oh spilly crocerdile saith he you tought that thear were kreetures that kept theeir haerst in a treetope you are a foole and i hav outvited you you may kep your friut to yoreself yore body is greuat but you hav no sesne

and thenn to eksplain ths ideya he luttered the followin stanzaz

rose-apfle yack-friute mnageso toos akrosse the watr thear i see
enouff of thm i wnt thm not my figg is goode henoufh for me
graet is yuor boddy verliy butt how muchh smaller is yoru witt
now go youre ways ser crocodile for i hve hdd th besst hof ith
the crocrdile feelingg as sadd and myserablle as if he had lost a housand pieses of muney wnt backk zorrowingk to the plase wher he livd
'''
    print(text_viterbi(typos, tree))
