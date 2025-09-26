# src/btx/scripts/format_beetlepalooza.py
"""
> tree -L 2 $HF_ROOT

> cat $HF_ROOT/BeetleMeasurements.csv | head
pictureID,scalebar,cm_pix,individual,structure,lying_flat,coords_pix,dist_pix,dist_cm,scientificName,NEON_sampleID,siteID,site_name,plotID,user_name,workflowID,genus,species,combinedID,measureID,file_name,image_dim,resized_image_dim,coords_pix_scaled_up
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,1,ElytraLength,Yes,"{""x1"": 1055, ""y1"": 154, ""x2"": 1163, ""y2"": 149}",108.115678788971,1.40409972453209,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_1,581c1309-6b06-4445-9ed5-55ebe366f6ed,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 3014, 'y1': 439, 'x2': 3323, 'y2': 425}"
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,1,ElytraWidth,Yes,"{""x1"": 1053, ""y1"": 129, ""x2"": 1057, ""y2"": 179}",50.1597448159378,0.651425257349842,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_1,464836fd-853e-40d5-861c-8c279aec6a55,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 3009, 'y1': 368, 'x2': 3020, 'y2': 511}"
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,2,ElytraLength,Yes,"{""x1"": 1390, ""y1"": 150, ""x2"": 1314, ""y2"": 241}",118.562219952226,1.53976909028865,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_2,80d48e56-c274-4ca9-854e-07605a62e140,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 3972, 'y1': 428, 'x2': 3754, 'y2': 688}"
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,2,ElytraWidth,Yes,"{""x1"": 1369, ""y1"": 136, ""x2"": 1407, ""y2"": 169}",50.3289181286465,0.653622313359045,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_2,839d9bde-1972-49d6-b31c-8aa81c84c0a2,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 3912, 'y1': 388, 'x2': 4020, 'y2': 482}"
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,3,ElytraLength,Yes,"{""x1"": 507, ""y1"": 378, ""x2"": 501, ""y2"": 487}",109.165012710117,1.41772743779373,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_3,d24c06fa-2779-45f9-8985-71c8e6e9418e,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 1448, 'y1': 1079, 'x2': 1431, 'y2': 1391}"
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,3,ElytraWidth,Yes,"{""x1"": 481, ""y1"": 378, ""x2"": 533, ""y2"": 381}",52.0864665724217,0.676447617823658,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_3,f6e635e4-ed4f-4b16-bd87-e98c2ac02812,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 1374, 'y1': 1079, 'x2': 1523, 'y2': 1088}"
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,4,ElytraLength,Yes,"{""x1"": 749, ""y1"": 381, ""x2"": 749, ""y2"": 486}",105.0,1.36363636363636,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_4,02c24662-60b6-4e28-a8b8-c2aa215ad45f,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 2140, 'y1': 1088, 'x2': 2140, 'y2': 1388}"
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,4,ElytraWidth,Yes,"{""x1"": 724, ""y1"": 383, ""x2"": 774, ""y2"": 382}",50.0099990001999,0.649480506496103,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_4,5b74abcf-1bf5-4b69-a45b-16c03ef881e2,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 2068, 'y1': 1094, 'x2': 2211, 'y2': 1091}"
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,5,ElytraLength,Yes,"{""x1"": 978, ""y1"": 417, ""x2"": 996, ""y2"": 519}",103.576059009792,1.3451436235038,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_5,3ff6fd36-e267-4c56-abfc-fca7822e76ea,group_images/A00000032929.jpg,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 2794, 'y1': 1191, 'x2': 2846, 'y2': 1482}"
> cat $HF_ROOT/individual_metadata.csv | head
individualID,combinedID,lying_flat,elytraLength,elytraWidth,measureID_length,measureID_width,genus,species,NEON_sampleID,file_name
581c1309-6b06-4445-9ed5-55ebe366f6ed_464836fd-853e-40d5-861c-8c279aec6a55,A00000032929_1,Yes,1.40409972453209,0.651425257349842,581c1309-6b06-4445-9ed5-55ebe366f6ed,464836fd-853e-40d5-861c-8c279aec6a55,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/581c1309-6b06-4445-9ed5-55ebe366f6ed_464836fd-853e-40d5-861c-8c279aec6a55.jpg
80d48e56-c274-4ca9-854e-07605a62e140_839d9bde-1972-49d6-b31c-8aa81c84c0a2,A00000032929_2,Yes,1.53976909028865,0.653622313359045,80d48e56-c274-4ca9-854e-07605a62e140,839d9bde-1972-49d6-b31c-8aa81c84c0a2,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/80d48e56-c274-4ca9-854e-07605a62e140_839d9bde-1972-49d6-b31c-8aa81c84c0a2.jpg
d24c06fa-2779-45f9-8985-71c8e6e9418e_f6e635e4-ed4f-4b16-bd87-e98c2ac02812,A00000032929_3,Yes,1.41772743779373,0.676447617823658,d24c06fa-2779-45f9-8985-71c8e6e9418e,f6e635e4-ed4f-4b16-bd87-e98c2ac02812,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/d24c06fa-2779-45f9-8985-71c8e6e9418e_f6e635e4-ed4f-4b16-bd87-e98c2ac02812.jpg
02c24662-60b6-4e28-a8b8-c2aa215ad45f_5b74abcf-1bf5-4b69-a45b-16c03ef881e2,A00000032929_4,Yes,1.36363636363636,0.649480506496103,02c24662-60b6-4e28-a8b8-c2aa215ad45f,5b74abcf-1bf5-4b69-a45b-16c03ef881e2,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/02c24662-60b6-4e28-a8b8-c2aa215ad45f_5b74abcf-1bf5-4b69-a45b-16c03ef881e2.jpg
3ff6fd36-e267-4c56-abfc-fca7822e76ea_6e982494-497f-4173-895d-b0a0a6d6c5d6,A00000032929_5,Yes,1.3451436235038,0.617122304961908,3ff6fd36-e267-4c56-abfc-fca7822e76ea,6e982494-497f-4173-895d-b0a0a6d6c5d6,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/3ff6fd36-e267-4c56-abfc-fca7822e76ea_6e982494-497f-4173-895d-b0a0a6d6c5d6.jpg
08e984aa-1592-4bd8-a0c8-3cd9a4d058b7_134061f6-254d-4010-9232-fd9cd1c63451,A00000032929_6,Yes,1.46804955836608,0.688434196399926,08e984aa-1592-4bd8-a0c8-3cd9a4d058b7,134061f6-254d-4010-9232-fd9cd1c63451,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/08e984aa-1592-4bd8-a0c8-3cd9a4d058b7_134061f6-254d-4010-9232-fd9cd1c63451.jpg
2fb0628e-b7bc-42eb-9d90-7c81821d88f2_f253372a-a8b4-40bc-a2fb-70e0423991af,A00000032929_7,Yes,1.42040166877464,0.70429853833539,2fb0628e-b7bc-42eb-9d90-7c81821d88f2,f253372a-a8b4-40bc-a2fb-70e0423991af,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/2fb0628e-b7bc-42eb-9d90-7c81821d88f2_f253372a-a8b4-40bc-a2fb-70e0423991af.jpg
766e92dc-3ca8-451d-ba7d-c89bafc863ca_4b9accad-4ec2-48fa-9d9f-5bbb95a48e9c,A00000032929_8,Yes,1.35064935064935,0.662337662337662,766e92dc-3ca8-451d-ba7d-c89bafc863ca,4b9accad-4ec2-48fa-9d9f-5bbb95a48e9c,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/766e92dc-3ca8-451d-ba7d-c89bafc863ca_4b9accad-4ec2-48fa-9d9f-5bbb95a48e9c.jpg
953e2fe7-ee75-4106-8c3a-6f26e192142b_66d15476-ff06-417c-99bb-6e4874bd8387,A00000032929_9,No,1.27100333814642,0.662846761307784,953e2fe7-ee75-4106-8c3a-6f26e192142b,66d15476-ff06-417c-99bb-6e4874bd8387,Carabus,goryi,HARV_001.20180605.CARGOR.01,individual_images/953e2fe7-ee75-4106-8c3a-6f26e192142b_66d15476-ff06-417c-99bb-6e4874bd8387.jpg
> cat $HF_ROOT/individual_specimens.csv | head
individualImageFilePath,groupImageFilePath,NEON_sampleID,scientificName,siteID,site_name,plotID
individual_specimens/part_000/A00000001831_specimen_1.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
individual_specimens/part_000/A00000001831_specimen_10.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
individual_specimens/part_000/A00000001831_specimen_11.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
individual_specimens/part_000/A00000001831_specimen_12.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
individual_specimens/part_000/A00000001831_specimen_13.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
individual_specimens/part_000/A00000001831_specimen_14.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
individual_specimens/part_000/A00000001831_specimen_15.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
individual_specimens/part_000/A00000001831_specimen_16.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
individual_specimens/part_000/A00000001831_specimen_17.png,group_images/A00000001831.jpg,SERC_010.20180523.CHLAES.01,Chlaenius aestivus,SERC,Smithsonian Environmental Research Center Site,SERC_010
> cat $HF_ROOT/group_images/metadata.csv | head
pictureID,scalebar,cm_pix,individual,structure,lying_flat,coords_pix,dist_pix,dist_cm,scientificName,NEON_sampleID,siteID,site_name,plotID,user_name,workflowID,genus,species,combinedID,measureID,image_dim,resized_image_dim,coords_pix_scaled_up,file_name,md5,subset
A00000032929.jpg,"{""x1"": 815, ""y1"": 244, ""x2"": 892, ""y2"": 244}",77.0,1,ElytraLength,Yes,"{""x1"": 1055, ""y1"": 154, ""x2"": 1163, ""y2"": 149}",108.115678788971,1.40409972453209,Carabus goryi,HARV_001.20180605.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Carabus,goryi,A00000032929_1,581c1309-6b06-4445-9ed5-55ebe366f6ed,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 3014, 'y1': 439, 'x2': 3323, 'y2': 425}",A00000032929.jpg,e2110ecefbf13d48f20fb4a51c6ff5a9,group_images
A00000033585.jpg,"{""x1"": 781, ""y1"": 253, ""x2"": 891, ""y2"": 254}",110.004545360635,1,ElytraLength,Yes,"{""x1"": 1010, ""y1"": 172, ""x2"": 1029, ""y2"": 229}",60.0832755431992,0.54618902651908,Synuchus impunctatus,HARV_008.20180814.SYNIMP.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_008,ishachinniah,21652,Synuchus,impunctatus,A00000033585_1,ea4658bb-c7fe-49b5-b3e7-c4c42aa8e084,"(3712, 5568, 3)","(1336, 2004, 3)","{'x1': 2806, 'y1': 477, 'x2': 2859, 'y2': 636}",A00000033585.jpg,37a7fb236e3e557fad52d3d5c1f6f054,group_images
HARV_008.W.20180814.PTEROS.01.jpg,"{""x1"": 898, ""y1"": 262, ""x2"": 1008, ""y2"": 263}",110.004545360635,1,ElytraLength,Yes,"{""x1"": 719, ""y1"": 429, ""x2"": 712, ""y2"": 516}",87.2811548961172,0.793432258730562,Pterostichus adoxus,HARV_008.W.20180814.PTEROS.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_008,ishachinniah,21652,Pterostichus,adoxus,HARV_008.W.20180814.PTEROS.01_1,bed56cab-5a55-4fda-924b-fa487731e9c8,"(3712, 5568, 3)","(1411, 2116, 3)","{'x1': 1891, 'y1': 1128, 'x2': 1873, 'y2': 1357}",HARV_008.W.20180814.PTEROS.01.jpg,bc29af5ff2eb65f1ddc1b769897e30a0,group_images
A00000033652.jpg,"{""x1"": 933, ""y1"": 298, ""x2"": 1056, ""y2"": 295}",123.036579926459,1,ElytraLength,Yes,"{""x1"": 1182, ""y1"": 199, ""x2"": 1258, ""y2"": 199}",76.0,0.617702475519283,Synuchus impunctatus,HARV_006.20180911.SYNIMP.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_006,IsaFluck,21652,Synuchus,impunctatus,A00000033652_1,a4ca6d1e-5c4e-45fa-8490-9eb20e3110a9,"(3712, 5568, 3)","(1336, 2004, 3)","{'x1': 3284, 'y1': 552, 'x2': 3495, 'y2': 552}",A00000033652.jpg,b1e79478c51643e71f217fdb50df66cc,group_images
A00000032963.jpg,"{""x1"": 476, ""y1"": 210, ""x2"": 554, ""y2"": 211}",78.0064099930256,1,ElytraLength,Yes,"{""x1"": 680, ""y1"": 121, ""x2"": 667, ""y2"": 240}",119.707978013163,1.53459155502562,Carabus goryi,HARV_005.20180814.CARGOR.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_005,IsaFluck,21652,Carabus,goryi,A00000032963_1,24912a13-cb37-47d4-bb64-637dc947739f,"(3712, 5568, 3)","(1188, 1782, 3)","{'x1': 2124, 'y1': 378, 'x2': 2084, 'y2': 749}",A00000032963.jpg,922b91bf8d70786d5e53f5880e925d79,group_images
A00000033686.jpg,"{""x1"": 929, ""y1"": 291, ""x2"": 1035, ""y2"": 291}",106.0,1,ElytraLength,Yes,"{""x1"": 555, ""y1"": 394, ""x2"": 558, ""y2"": 454}",60.0749531835024,0.566744841353796,Synuchus impunctatus,HARV_005.20180814.SYNIMP.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_005,IsaFluck,21652,Synuchus,impunctatus,A00000033686_1,e311e80b-3e16-4a0e-9b06-b3f8496623ab,"(3712, 5568, 3)","(1336, 2004, 3)","{'x1': 1542, 'y1': 1094, 'x2': 1550, 'y2': 1261}",A00000033686.jpg,1f428e3514d3899db52f36ea16801911,group_images
A00000033635.jpg,"{""x1"": 929, ""y1"": 271, ""x2"": 1033, ""y2"": 271}",104.0,1,ElytraLength,Yes,"{""x1"": 548, ""y1"": 433, ""x2"": 549, ""y2"": 500}",67.0074622710038,0.644302521836575,Synuchus impunctatus,HARV_001.20180703.SYNIMP.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,ishachinniah,21652,Synuchus,impunctatus,A00000033635_1,870887d6-34a1-481e-8787-c9581b048298,"(3712, 5568, 3)","(1336, 2004, 3)","{'x1': 1522, 'y1': 1203, 'x2': 1525, 'y2': 1389}",A00000033635.jpg,91f21036d2a90886d06837d267468193,group_images
A00000033601.jpg,"{""x1"": 766, ""y1"": 218, ""x2"": 870, ""y2"": 219}",104.004807581188,1,ElytraLength,Yes,"{""x1"": 1088, ""y1"": 144, ""x2"": 1089, ""y2"": 203}",59.0084739677277,0.567362945426003,Synuchus impunctatus,HARV_008.20180828.SYNIMP.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_008,rileywolcheski,21652,Synuchus,impunctatus,A00000033601_1,61416389-7209-47a0-b4ca-a7d9fc73c0d8,"(3712, 5568, 3)","(1299, 1949, 3)","{'x1': 3109, 'y1': 411, 'x2': 3111, 'y2': 579}",A00000033601.jpg,c05fde8e23f806044087fff7ffbde5e5,group_images
A00000033618.jpg,"{""x1"": 808, ""y1"": 264, ""x2"": 914, ""y2"": 265}",106.004716876184,1,ElytraLength,Yes,"{""x1"": 538, ""y1"": 387, ""x2"": 537, ""y2"": 453}",66.007575322837,0.622685265976751,Synuchus impunctatus,HARV_001.20180828.SYNIMP.01,HARV,Harvard Forest & Quabbin Watershed NEON,HARV_001,IsaFluck,21652,Synuchus,impunctatus,A00000033618_1,81002603-c4f5-4190-89c4-409aecb4f2ca,"(3712, 5568, 3)","(1336, 2004, 3)","{'x1': 1494, 'y1': 1075, 'x2': 1492, 'y2': 1258}",A00000033618.jpg,996389657f96326febc72198be10f210,group_images
"""

import ast
import dataclasses
import gc
import json
import logging
import pathlib
import resource
import time
import typing as tp

import beartype
import numpy as np
import polars as pl
import skimage.feature
import submitit
import tyro
from jaxtyping import Float, UInt
from PIL import Image, ImageDraw

import btx.helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    hf_root: pathlib.Path = pathlib.Path("./data/beetlepalooza")
    """Where you dumped data when using download_beetlepalooza.py."""

    resized_root: pathlib.Path = pathlib.Path("./data/beetlepalooza-resized-images")
    """I don't know why the huggingface dataset doesn't have the right images. I don't make the rules. I just have to play by them."""

    log_to: pathlib.Path = pathlib.Path("./logs")
    """Where to save submitit/slurm logs."""

    dump_to: pathlib.Path = pathlib.Path("./data/beetlepalooza-formatted")
    """Where to save formatted data."""

    ignore_errors: bool = False
    """Skip the user error check and always proceed (equivalent to answering 'yes')."""

    seed: int = 42
    """Random seed for sampling which annotations to save as examples."""

    sample_rate: int = 20
    """Save 1 in sample_rate annotations as example images (default: 1 in 20)."""

    # Slurm configuration
    slurm_acct: str = ""
    """Slurm account to use. If empty, uses DebugExecutor."""

    slurm_partition: str = "parallel"
    """Slurm partition to use."""

    n_hours: float = 2.0
    """Number of hours to request for each job."""

    groups_per_job: int = 4
    """Number of group images to process per job."""


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ValidationError:
    """Data validation error with type and count."""

    error_type: tp.Literal[
        "specimen_duplicates",
        "missing_files",
        "corrupted_files",
        "dimension_errors",
        "images_without_measurements",
        "measurements_without_images",
    ]
    count: int
    details: list[str] = dataclasses.field(default_factory=list)
    """Optional list of example error details for logging."""

    def log_summary(self, logger: logging.Logger) -> None:
        """Log a summary of this error."""
        logger.error("Found %d %s", self.count, self.error_type.replace("_", " "))
        for detail in self.details[:10]:  # Show first 10 examples
            logger.error("  %s", detail)
        if len(self.details) > 10:
            logger.error("  ... and %d more", len(self.details) - 10)

    @property
    def display_name(self) -> str:
        """Human-readable error type name."""
        return self.error_type.replace("_", " ").title()


@beartype.beartype
def img_as_arr(
    img: Image.Image | pathlib.Path,
) -> Float[np.ndarray, "height width channels"]:
    img = img if isinstance(img, Image.Image) else Image.open(img)
    return np.asarray(img, dtype=np.float32)


@beartype.beartype
def img_as_grayscale(
    img: Image.Image | pathlib.Path,
) -> UInt[np.ndarray, "height width"]:
    img = img if isinstance(img, Image.Image) else Image.open(img)
    # Convert to grayscale using PIL (more efficient than loading RGB then converting)
    return np.asarray(img.convert("L"))


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class WorkerError:
    """Base class for worker errors with context."""

    group_img_basename: str
    message: str

    def __str__(self):
        return f"{self.group_img_basename}: {self.message}"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class TemplateMatchingError(WorkerError):
    """Error during template matching."""

    beetle_position: int
    indiv_img_path: str

    def __str__(self):
        return f"{self.group_img_basename} (beetle {self.beetle_position}): Template matching failed for {self.indiv_img_path} - {self.message}"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class ImageLoadError(WorkerError):
    """Error loading an image file."""

    img_path: str

    def __str__(self):
        return f"{self.group_img_basename}: Failed to load {self.img_path} - {self.message}"


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Annotation:
    group_img_basename: str
    beetle_position: int
    group_img_abs_path: pathlib.Path
    indiv_img_abs_path: pathlib.Path
    indiv_offset_px: tuple[float, float]
    individual_id: str
    ncc: float
    """Normalized cross-correlation score from template matching."""
    neon_sample_id: str
    """NEON sample ID."""
    scientific_name: str
    """Scientific name (genus species)."""

    def to_dict(self) -> dict:
        """Convert annotation to dictionary for JSON serialization."""
        return {
            "group_img_basename": self.group_img_basename,
            "beetle_position": self.beetle_position,
            "group_img_rel_path": f"group_images/{self.group_img_basename}",
            "indiv_img_rel_path": str(self.indiv_img_abs_path).split("/beetlepalooza/")[
                -1
            ]
            if "/beetlepalooza/" in str(self.indiv_img_abs_path)
            else str(self.indiv_img_abs_path.name),
            "indiv_img_abs_path": str(self.indiv_img_abs_path),
            "individual_id": self.individual_id,
            "origin_x": int(self.indiv_offset_px[0]),
            "origin_y": int(self.indiv_offset_px[1]),
            "ncc": self.ncc,
            "neon_sample_id": self.neon_sample_id,
            "scientific_name": self.scientific_name,
        }


@beartype.beartype
def save_example_images(
    dump_to: pathlib.Path, annotation: Annotation, measurement_data: dict[str, object]
) -> None:
    """Save example images with annotations drawn on them."""
    # Define colors for different measurement types (RGB)
    measurement_colors = {
        "ElytraLength": (0, 255, 0),  # Green
        "ElytraWidth": (255, 255, 0),  # Yellow
        "PronotumWidth": (0, 0, 255),  # Blue
    }
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("save-imgs")

    # Load images for drawing
    try:
        group_img_pil = Image.open(annotation.group_img_abs_path).convert("RGB")
        indiv_img_pil = Image.open(annotation.indiv_img_abs_path).convert("RGB")
    except Exception as e:
        logger.warning(
            "Failed to load images for example: %s beetle %d - %s",
            annotation.group_img_basename,
            annotation.beetle_position,
            e,
        )
        return

    # Get individual image dimensions
    indiv_w, indiv_h = indiv_img_pil.size
    x, y = annotation.indiv_offset_px

    # Draw on group image
    group_draw = ImageDraw.Draw(group_img_pil)

    # Draw bounding box around individual beetle
    group_draw.rectangle(
        (x, y, x + indiv_w, y + indiv_h),
        outline=(255, 0, 0),
        width=12,
    )

    # Draw lines for each measurement type on group image
    for structure, color in measurement_colors.items():
        if structure not in measurement_data:
            continue

        coords = measurement_data[structure]
        if not coords:
            continue

        # coords is a dict with x1, y1, x2, y2
        try:
            group_draw.line(
                [(coords["x1"], coords["y1"]), (coords["x2"], coords["y2"])],
                fill=color,
                width=8,
            )
        except (KeyError, TypeError):
            logger.warning(
                "Invalid coords for %s on %s beetle %d",
                structure,
                annotation.group_img_basename,
                annotation.beetle_position,
            )
            continue

    # Draw on individual image with adjusted coordinates
    indiv_draw = ImageDraw.Draw(indiv_img_pil)

    for structure, color in measurement_colors.items():
        if structure not in measurement_data:
            continue

        coords = measurement_data[structure]
        if not coords:
            continue

        try:
            # Adjust coordinates relative to individual image
            adjusted_points = [
                (coords["x1"] - x, coords["y1"] - y),
                (coords["x2"] - x, coords["y2"] - y),
            ]

            # Check if points are within bounds
            valid_points = []
            for px, py in adjusted_points:
                if 0 <= px < indiv_w and 0 <= py < indiv_h:
                    valid_points.append((px, py))

            if len(valid_points) == 2:
                indiv_draw.line(valid_points, fill=color, width=3)
        except (KeyError, TypeError):
            continue

    # Resize group image for viewing
    group_w, group_h = group_img_pil.size
    resized_group = group_img_pil.resize((group_w // 10, group_h // 10))

    # Save images
    examples_dir = dump_to / "random-examples"
    group_path = (
        examples_dir
        / f"{annotation.group_img_basename.replace('.jpg', '')}_beetle{annotation.beetle_position}_group.png"
    )
    indiv_path = (
        examples_dir
        / f"{annotation.group_img_basename.replace('.jpg', '')}_beetle{annotation.beetle_position}_individual.png"
    )

    resized_group.save(group_path)
    indiv_img_pil.save(indiv_path)

    logger.info(
        "Saved example images for %s beetle %d",
        annotation.group_img_basename,
        annotation.beetle_position,
    )


@beartype.beartype
def get_memory_info() -> dict[str, float]:
    """Get current memory usage information."""
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        meminfo = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                value = int(parts[1])
                meminfo[key] = value

        mem_total_gb = meminfo.get("MemTotal", 0) / (1024 * 1024)
        mem_available_gb = meminfo.get("MemAvailable", 0) / (1024 * 1024)
        mem_free_gb = meminfo.get("MemFree", 0) / (1024 * 1024)

        # Also get process-specific memory
        usage = resource.getrusage(resource.RUSAGE_SELF)
        process_mem_gb = usage.ru_maxrss / (1024 * 1024)  # Linux reports in KB

        return {
            "total_gb": round(mem_total_gb, 2),
            "available_gb": round(mem_available_gb, 2),
            "free_gb": round(mem_free_gb, 2),
            "used_gb": round(mem_total_gb - mem_available_gb, 2),
            "process_gb": round(process_mem_gb, 2),
            "percent_used": round(
                (mem_total_gb - mem_available_gb) / mem_total_gb * 100, 1
            ),
        }
    except Exception as e:
        return {"error": str(e)}


@beartype.beartype
def worker_fn(
    cfg: Config, group_img_basenames: list[str]
) -> list[Annotation | WorkerError]:
    """Worker. Processing group_img_basenames and returns a list of annotations or errors."""
    logging.basicConfig(level=logging.DEBUG, format=log_format)
    logger = logging.getLogger("worker")

    # Log initial memory state
    mem_info = get_memory_info()
    logger.info(
        "Starting worker with %d group images. Memory: %s",
        len(group_img_basenames),
        mem_info,
    )

    # Load dataframes
    specimens_df = load_specimens_df(cfg)
    logger.info(
        "Loaded specimens_df with %d rows. Memory: %s",
        len(specimens_df),
        get_memory_info(),
    )

    measurements_df = load_measurements_df(cfg)
    logger.info(
        "Loaded measurements_df with %d rows. Memory: %s",
        len(measurements_df),
        get_memory_info(),
    )

    # Filter to only relevant data to save memory
    logger.info("Filtering dataframes to relevant groups...")
    specimens_df = specimens_df.filter(
        pl.col("GroupImgBasename").is_in(group_img_basenames)
    )
    measurements_df = measurements_df.filter(
        pl.col("GroupImgBasename").is_in(group_img_basenames)
    )
    logger.info(
        "After filtering - specimens_df: %d rows, measurements_df: %d rows. Memory: %s",
        len(specimens_df),
        len(measurements_df),
        get_memory_info(),
    )

    results = []

    # Initialize random number generator for sampling
    rng = np.random.default_rng(seed=cfg.seed)

    for idx, group_img_basename in enumerate(group_img_basenames):
        logger.info(
            "Processing group %d/%d: %s. Memory before: %s",
            idx + 1,
            len(group_img_basenames),
            group_img_basename,
            get_memory_info(),
        )

        # Construct the group image path
        group_img_abs_path = cfg.resized_root / group_img_basename

        # Load the group image in grayscale for matching.
        try:
            group_img_gray = img_as_grayscale(group_img_abs_path)
            logger.info(
                "Loaded group image %s, gray shape: %s. Memory: %s",
                group_img_basename,
                group_img_gray.shape,
                get_memory_info(),
            )
        except Exception as e:
            logger.error("Failed to load group image %s: %s", group_img_basename, e)
            results.append(
                ImageLoadError(
                    group_img_basename=group_img_basename,
                    message=str(e),
                    img_path=str(group_img_abs_path),
                )
            )
            continue

        # Find all individual images for this group image
        group_rows = specimens_df.filter(
            pl.col("GroupImgBasename") == group_img_basename
        )
        logger.info(
            "Found %d individual images for group %s",
            len(group_rows),
            group_img_basename,
        )

        for row in group_rows.iter_rows(named=True):
            beetle_position = row["BeetlePosition"]
            indiv_img_rel_path = row["individualImageFilePath"]
            indiv_img_abs_path = cfg.hf_root / indiv_img_rel_path

            # Load the individual image (grayscale for matching)
            try:
                template_gray = img_as_grayscale(indiv_img_abs_path)
                logger.debug(
                    "Loaded individual image for beetle %d, gray shape: %s. Memory: %s",
                    beetle_position,
                    template_gray.shape,
                    get_memory_info(),
                )
            except Exception as e:
                logger.error(
                    "Failed to load individual image for beetle %d: %s",
                    beetle_position,
                    e,
                )
                results.append(
                    ImageLoadError(
                        group_img_basename=group_img_basename,
                        message=str(e),
                        img_path=str(indiv_img_abs_path),
                    )
                )
                continue

            # Perform template matching (using grayscale images)
            try:
                logger.debug(
                    "Starting template matching for beetle %d...", beetle_position
                )
                corr = skimage.feature.match_template(
                    group_img_gray, template_gray, pad_input=False
                )
                logger.debug(
                    "Template matching complete, corr shape: %s. Memory: %s",
                    corr.shape,
                    get_memory_info(),
                )

                if corr.size == 0:
                    raise ValueError(
                        "Empty correlation map - template may be larger than image"
                    )

                max_corr_idx = np.argmax(corr)
                iy, ix = np.unravel_index(max_corr_idx, corr.shape)
                offset_px = float(ix), float(iy)

                # Get the normalized cross-correlation score at the best match
                ncc_score = float(corr.flat[max_corr_idx])

                # Clean up correlation matrix to free memory
                del corr
                del template_gray
                gc.collect()

                # Get metadata from the row data
                individual_id = (
                    f"{group_img_basename.replace('.jpg', '')}_{beetle_position}"
                )
                neon_sample_id = row.get("NEON_sampleID", "")
                scientific_name = row.get("scientificName", "")

                annotation = Annotation(
                    group_img_basename=group_img_basename,
                    beetle_position=beetle_position,
                    group_img_abs_path=group_img_abs_path,
                    indiv_img_abs_path=indiv_img_abs_path,
                    indiv_offset_px=offset_px,
                    individual_id=individual_id,
                    ncc=ncc_score,
                    neon_sample_id=neon_sample_id,
                    scientific_name=scientific_name,
                )

                # Save a random subset of annotations as example images
                if rng.integers(0, cfg.sample_rate) == 0:
                    measurement_rows = measurements_df.filter(
                        (pl.col("GroupImgBasename") == annotation.group_img_basename)
                        & (pl.col("BeetlePosition") == annotation.beetle_position)
                    )

                    if not measurement_rows.is_empty():
                        measurement_data = {}
                        for mrow in measurement_rows.iter_rows(named=True):
                            structure = mrow.get("structure", "")
                            coords = mrow.get("coords_pix", {})
                            if structure and coords:
                                measurement_data[structure] = coords

                        if measurement_data:
                            save_example_images(
                                cfg.dump_to, annotation, measurement_data
                            )
                            logger.debug(
                                "Saved example image. Memory: %s", get_memory_info()
                            )

                results.append(annotation)

            except Exception as e:
                results.append(
                    TemplateMatchingError(
                        group_img_basename=group_img_basename,
                        message=str(e),
                        beetle_position=beetle_position,
                        indiv_img_path=str(indiv_img_abs_path),
                    )
                )

        # Clean up group images after processing all individuals
        del group_img_gray
        gc.collect()
        logger.info(
            "Finished processing group %s. Memory after cleanup: %s",
            group_img_basename,
            get_memory_info(),
        )

    logger.info(
        "Worker completed. Processed %d groups, %d results. Final memory: %s",
        len(group_img_basenames),
        len(results),
        get_memory_info(),
    )
    return results


@beartype.beartype
def load_measurements_df(cfg: Config) -> pl.DataFrame:
    """Load and process the BeetleMeasurements.csv file."""
    df = pl.read_csv(cfg.hf_root / "BeetleMeasurements.csv")

    # Parse the coords_pix JSON string
    def parse_coords(coords_str):
        if coords_str:
            try:
                # Handle both single and double quotes
                coords_str = coords_str.replace('""', '"')
                return ast.literal_eval(coords_str)
            except (ValueError, SyntaxError):
                return {}
        return {}

    # Process dataframe
    df = df.with_columns(
        pl.col("pictureID").alias("GroupImgBasename"),
        pl.col("individual").alias("BeetlePosition"),
        pl.col("coords_pix").map_elements(parse_coords, return_dtype=pl.Object),
    )

    return df


@beartype.beartype
def load_specimens_df(cfg: Config) -> pl.DataFrame:
    """Load and process the individual_specimens.csv file."""
    df = pl.read_csv(cfg.hf_root / "individual_specimens.csv")

    # Extract beetle position from the individual image filename
    # e.g., "A00000001831_specimen_1.png" -> 1
    df = df.with_columns(
        pl.col("groupImageFilePath")
        .str.strip_prefix("group_images/")
        .alias("GroupImgBasename"),
        pl.col("individualImageFilePath")
        .str.extract(r"specimen_(\d+)", 1)
        .cast(pl.Int64)
        .alias("BeetlePosition"),
    )

    return df


@beartype.beartype
def validate_data(
    cfg: Config, specimens_df: pl.DataFrame, measurements_df: pl.DataFrame
) -> list[ValidationError]:
    """Validate the data and return a list of validation errors."""
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("format-bp")
    errors: list[ValidationError] = []

    # Check for duplicate specimens
    specimen_dups = (
        specimens_df.group_by("GroupImgBasename", "BeetlePosition")
        .len()
        .filter(pl.col("len") > 1)
    )
    if not specimen_dups.is_empty():
        details = []
        for row in specimen_dups.iter_rows(named=True):
            details.append(
                f"GroupImgBasename: {row['GroupImgBasename']}, "
                f"BeetlePosition: {row['BeetlePosition']}, count: {row['len']}"
            )
        error = ValidationError(
            error_type="specimen_duplicates",
            count=len(specimen_dups),
            details=details,
        )
        error.log_summary(logger)
        errors.append(error)

    # Check file existence and corruption
    logger.info("Checking that image files exist.")
    all_group_paths = set(specimens_df["groupImageFilePath"].unique().to_list())
    all_individual_paths = set(
        specimens_df["individualImageFilePath"].unique().to_list()
    )
    all_paths = all_group_paths | all_individual_paths

    missing_files = []
    corrupted_files = []

    for i, rel_path in enumerate(
        btx.helpers.progress(
            sorted(all_paths), every=len(all_paths) // 10, desc="file existence"
        )
    ):
        # Check if this is a group image path (starts with "group_images/")
        if rel_path in all_group_paths:
            # For group images, use resized_root with just the basename
            basename = rel_path.split("/")[-1]
            full_path = cfg.resized_root / basename
        else:
            # For individual images, use hf_root with the full relative path
            full_path = cfg.hf_root / rel_path

        if not full_path.exists():
            missing_files.append(str(rel_path))
        else:
            # Try to open the file to check if it's corrupted
            try:
                with Image.open(full_path) as img:
                    _ = img.size  # Force loading to check corruption
            except Exception as e:
                corrupted_files.append(f"{rel_path}: {e}")

    if missing_files:
        error = ValidationError(
            error_type="missing_files",
            count=len(missing_files),
            details=missing_files,
        )
        error.log_summary(logger)
        errors.append(error)

    if corrupted_files:
        error = ValidationError(
            error_type="corrupted_files",
            count=len(corrupted_files),
            details=corrupted_files,
        )
        error.log_summary(logger)
        errors.append(error)

    # Check image dimensions
    logger.info("Checking image dimensions.")
    group_to_individuals = (
        specimens_df.group_by("groupImageFilePath")
        .agg(pl.col("individualImageFilePath").unique())
        .to_dicts()
    )

    dimension_error_details = []
    for group_data in btx.helpers.progress(
        group_to_individuals,
        every=len(group_to_individuals) // 10,
        desc="dimension check",
    ):
        # Extract basename from path like "group_images/A00000001831.jpg"
        group_basename = group_data["groupImageFilePath"].split("/")[-1]
        group_path = cfg.resized_root / group_basename

        # Skip if file doesn't exist (already reported)
        if not group_path.exists():
            continue

        try:
            with Image.open(group_path) as group_img:
                group_width, group_height = group_img.size

                for individual_rel_path in group_data["individualImageFilePath"]:
                    individual_path = cfg.hf_root / individual_rel_path

                    # Skip if file doesn't exist (already reported)
                    if not individual_path.exists():
                        continue

                    try:
                        with Image.open(individual_path) as ind_img:
                            ind_width, ind_height = ind_img.size

                            if ind_width >= group_width or ind_height >= group_height:
                                dimension_error_details.append(
                                    f"Individual {individual_rel_path} ({ind_width}x{ind_height}) >= "
                                    f"Group {group_data['groupImageFilePath']} ({group_width}x{group_height})"
                                )
                    except Exception:
                        # Skip corrupted files (already reported)
                        pass

        except Exception:
            # Skip corrupted files (already reported)
            pass

    if dimension_error_details:
        error = ValidationError(
            error_type="dimension_errors",
            count=len(dimension_error_details),
            details=dimension_error_details,
        )
        error.log_summary(logger)
        errors.append(error)

    return errors


@beartype.beartype
def handle_validation_errors(errors: list[ValidationError], cfg: Config) -> bool:
    """Handle validation errors. Returns True if should continue, False if should exit."""
    logger = logging.getLogger("format-bp")

    if not errors:
        logger.info("No data validation errors found.")
        return True

    print("\n" + "=" * 60)
    print("DATA VALIDATION SUMMARY")
    print("=" * 60)
    print("\nThe following issues were found:")
    for error in errors:
        print(f"  - {error.display_name}: {error.count}")

    print("\n" + "=" * 60)

    if cfg.ignore_errors:
        logger.warning("Ignoring errors due to --ignore-errors flag. Continuing.")
        return True

    response = input(
        "\nDo you want to continue with template matching despite these errors? (yes/no): "
    )
    if response.lower() not in ["yes", "y"]:
        logger.info("Exiting.")
        return False
    logger.info("Continuing.")
    return True


@beartype.beartype
def setup_executor(cfg: Config, n_batches: int) -> tuple[submitit.Executor, int, int]:
    """Set up the executor (Slurm or Debug) and return executor, safe_array_size, safe_submit_jobs."""
    logger = logging.getLogger("format-bp")

    if cfg.slurm_acct:
        # Calculate safe limits for Slurm
        max_array_size = btx.helpers.get_slurm_max_array_size()
        max_submit_jobs = btx.helpers.get_slurm_max_submit_jobs()

        safe_array_size = min(int(max_array_size * 0.95), max_array_size - 2)
        safe_array_size = max(1, safe_array_size)

        safe_submit_jobs = min(int(max_submit_jobs * 0.95), max_submit_jobs - 2)
        safe_submit_jobs = max(1, safe_submit_jobs)

        logger.info(
            "Using Slurm with safe limits - Array size: %d (max: %d), Submit jobs: %d (max: %d)",
            safe_array_size,
            max_array_size,
            safe_submit_jobs,
            max_submit_jobs,
        )

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=int(cfg.n_hours * 60),  # Convert hours to minutes
            partition=cfg.slurm_partition,
            gpus_per_node=0,
            ntasks_per_node=1,
            cpus_per_task=8,
            mem_per_cpu="10gb",
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
            array_parallelism=safe_array_size,
        )
    else:
        logger.info("Using DebugExecutor for local execution")
        executor = submitit.DebugExecutor(folder=cfg.log_to)
        safe_array_size = n_batches
        safe_submit_jobs = n_batches

    return executor, safe_array_size, safe_submit_jobs


@beartype.beartype
def collect_results(jobs: list) -> tuple[list[Annotation], list[WorkerError]]:
    """Collect results from all jobs and return annotations and errors."""
    logger = logging.getLogger("format-bp")
    all_annotations = []
    all_errors = []

    for job_idx, job in enumerate(jobs):
        try:
            results = job.result()
            for result in results:
                if isinstance(result, WorkerError):
                    all_errors.append(result)
                else:
                    all_annotations.append(result)
            logger.info("Job %d/%d completed", job_idx + 1, len(jobs))
        except Exception as e:
            logger.error("Job %d/%d failed: %s", job_idx + 1, len(jobs), e)

    return all_annotations, all_errors


@beartype.beartype
def report_statistics(
    annotations: list[Annotation], errors: list[WorkerError], expected_count: int
) -> None:
    """Report final statistics about processing."""
    logger = logging.getLogger("format-bp")

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info("Total annotations: %d", len(annotations))
    logger.info("Total errors: %d", len(errors))

    if errors:
        logger.info("\nError summary:")
        error_types = {}
        for error in errors:
            error_type = type(error).__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
        for error_type, count in error_types.items():
            logger.info("  %s: %d", error_type, count)

    # Check expected vs actual annotation count
    logger.info("\nExpected annotations: %d", expected_count)
    logger.info("Actual annotations: %d", len(annotations))
    if expected_count != len(annotations) + len(errors):
        logger.warning(
            "Mismatch in annotation count! Missing: %d",
            expected_count - len(annotations) - len(errors),
        )


@beartype.beartype
def save_annotations(
    cfg: Config, annotations: list[Annotation], measurements_df: pl.DataFrame
) -> None:
    """Save annotations to JSON file with measurements."""
    logger = logging.getLogger("format-bp")

    output_file = cfg.dump_to / "annotations.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving annotations to %s", output_file)

    # Convert annotations to JSON format with measurements
    json_data = []

    for annotation in annotations:
        # Get base annotation data
        ann_dict = annotation.to_dict()

        # Get measurements for this beetle
        measurement_rows = measurements_df.filter(
            (pl.col("GroupImgBasename") == annotation.group_img_basename)
            & (pl.col("BeetlePosition") == annotation.beetle_position)
        )

        # Add measurements if available
        measurements = []
        if not measurement_rows.is_empty():
            for row in measurement_rows.iter_rows(named=True):
                structure = row.get("structure", "")
                coords = row.get("coords_pix", {})
                dist_cm = row.get("dist_cm", None)

                if structure and coords and "x1" in coords:
                    # Adjust coordinates relative to individual image
                    origin_x, origin_y = annotation.indiv_offset_px
                    adjusted_coords = {
                        "x1": coords["x1"] - origin_x,
                        "y1": coords["y1"] - origin_y,
                        "x2": coords["x2"] - origin_x,
                        "y2": coords["y2"] - origin_y,
                    }

                    measurements.append({
                        "measurement_type": structure.lower().replace(
                            "elytra", "elytra_"
                        ),
                        "coords_px": adjusted_coords,
                        "dist_cm": dist_cm,
                    })

        ann_dict["measurements"] = measurements
        json_data.append(ann_dict)

    # Save to JSON file
    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)

    logger.info("Saved %d annotations to %s", len(json_data), output_file)


@beartype.beartype
def main(cfg: Config) -> int:
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger("format-bp")

    specimens_df = load_specimens_df(cfg)
    logger.info("Loaded specimens_df with %d rows", len(specimens_df))

    measurements_df = load_measurements_df(cfg)
    logger.info("Loaded measurements_df with %d rows", len(measurements_df))

    errors = validate_data(cfg, specimens_df, measurements_df)
    if not handle_validation_errors(errors, cfg):
        return 1

    logger.info("Ready for parallel processing implementation.")

    # Create output directory for example images
    examples_dir = cfg.dump_to / "random-examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Example images will be saved to %s", examples_dir)

    # Get all unique group images to process
    all_group_basenames = specimens_df.get_column("GroupImgBasename").unique().to_list()
    logger.info("Found %d unique group images to process", len(all_group_basenames))

    # Batch group images into chunks for each job
    group_batches = list(
        btx.helpers.batched_idx(len(all_group_basenames), cfg.groups_per_job)
    )
    logger.info(
        "Will process %d group images in %d jobs (%d groups per job)",
        len(all_group_basenames),
        len(group_batches),
        cfg.groups_per_job,
    )

    executor, safe_array_size, safe_submit_jobs = setup_executor(
        cfg, len(group_batches)
    )

    # Submit jobs in batches to respect Slurm limits
    all_jobs = []
    job_batches = list(btx.helpers.batched_idx(len(group_batches), safe_array_size))

    for batch_idx, (start, end) in enumerate(job_batches):
        current_batches = group_batches[start:end]

        # Check current job count and wait if needed (only for Slurm)
        if cfg.slurm_acct:
            current_jobs = btx.helpers.get_slurm_job_count()
            jobs_available = max(0, safe_submit_jobs - current_jobs)

            while jobs_available < len(current_batches):
                logger.info(
                    "Can only submit %d jobs but need %d. Waiting for jobs to complete...",
                    jobs_available,
                    len(current_batches),
                )
                time.sleep(60)  # Wait 1 minute
                current_jobs = btx.helpers.get_slurm_job_count()
                jobs_available = max(0, safe_submit_jobs - current_jobs)

        logger.info(
            "Submitting job batch %d/%d: jobs %d-%d",
            batch_idx + 1,
            len(job_batches),
            start,
            end - 1,
        )

        # Submit jobs for this batch
        with executor.batch():
            for group_start, group_end in current_batches:
                group_batch = all_group_basenames[group_start:group_end]
                job = executor.submit(worker_fn, cfg, group_batch)
                all_jobs.append(job)

        logger.info("Submitted job batch %d/%d", batch_idx + 1, len(job_batches))

    logger.info("Submitted %d total jobs. Waiting for results...", len(all_jobs))

    all_annotations, all_errors = collect_results(all_jobs)
    expected_count = len(specimens_df)
    report_statistics(all_annotations, all_errors, expected_count)
    if not all_annotations:
        return 1

    save_annotations(cfg, all_annotations, measurements_df)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(tyro.cli(Config)))
    except KeyboardInterrupt:
        print("Interrupted.")
        raise SystemExit(130)
