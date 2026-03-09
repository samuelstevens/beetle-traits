Module btx.scripts.format_beetlepalooza
=======================================
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

Functions
---------

`collect_results(jobs: list) ‑> tuple[list[btx.scripts.format_beetlepalooza.Annotation], list[btx.scripts.format_beetlepalooza.WorkerError]]`
:   Collect results from all jobs and return annotations and errors.

`get_memory_info() ‑> dict[str, float]`
:   Get current memory usage information.

`handle_validation_errors(errors: list[btx.scripts.format_beetlepalooza.ValidationError], cfg: btx.scripts.format_beetlepalooza.Config) ‑> bool`
:   Handle validation errors. Returns True if should continue, False if should exit.

`img_as_arr(img: PIL.Image.Image | pathlib.Path) ‑> jaxtyping.Float[ndarray, 'height width channels']`
:   

`img_as_grayscale(img: PIL.Image.Image | pathlib.Path) ‑> jaxtyping.UInt[ndarray, 'height width']`
:   

`load_measurements_df(cfg: btx.scripts.format_beetlepalooza.Config) ‑> polars.dataframe.frame.DataFrame`
:   Load and process the BeetleMeasurements.csv file.

`load_specimens_df(cfg: btx.scripts.format_beetlepalooza.Config) ‑> polars.dataframe.frame.DataFrame`
:   Load specimens mapping; supports old (individual_specimens.csv) and new split metadata.

`main(cfg: btx.scripts.format_beetlepalooza.Config) ‑> int`
:   

`report_statistics(annotations: list[btx.scripts.format_beetlepalooza.Annotation], errors: list[btx.scripts.format_beetlepalooza.WorkerError], expected_count: int) ‑> None`
:   Report final statistics about processing.

`run_position_correction(cfg: btx.scripts.format_beetlepalooza.Config, logger: logging.Logger) ‑> bool`
:   Run row_template_match_rename.py to correct beetle positions before template matching.
    
    Returns:
        True if successful or skipped, False if failed

`run_validation_cleanup(cfg: btx.scripts.format_beetlepalooza.Config, logger: logging.Logger) ‑> bool`
:   Run validate_beetlepalooza_annotations.py to remove invalid measurements.
    
    Returns:
        True if successful, False if failed

`save_annotations(cfg: btx.scripts.format_beetlepalooza.Config, annotations: list[btx.scripts.format_beetlepalooza.Annotation], measurements_df: polars.dataframe.frame.DataFrame) ‑> None`
:   Save annotations to JSON file with measurements.

`save_example_images(dump_to: pathlib.Path, annotation: btx.scripts.format_beetlepalooza.Annotation, measurement_data: dict[str, object]) ‑> None`
:   Save example images with annotations drawn on them.

`setup_executor(cfg: btx.scripts.format_beetlepalooza.Config, n_batches: int) ‑> tuple[submitit.core.core.Executor, int, int]`
:   Set up the executor (Slurm or Debug) and return executor, safe_array_size, safe_submit_jobs.

`validate_data(cfg: btx.scripts.format_beetlepalooza.Config, specimens_df: polars.dataframe.frame.DataFrame, measurements_df: polars.dataframe.frame.DataFrame) ‑> list[btx.scripts.format_beetlepalooza.ValidationError]`
:   Validate the data and return a list of validation errors.

`worker_fn(cfg: btx.scripts.format_beetlepalooza.Config, group_img_basenames: list[str]) ‑> list[btx.scripts.format_beetlepalooza.Annotation | btx.scripts.format_beetlepalooza.WorkerError]`
:   Worker. Processing group_img_basenames and returns a list of annotations or errors.

Classes
-------

`Annotation(group_img_basename: str, beetle_position: int, group_img_abs_path: pathlib.Path, indiv_img_abs_path: pathlib.Path, indiv_offset_px: tuple[float, float], individual_id: str, ncc: float, neon_sample_id: str, scientific_name: str)`
:   Annotation(group_img_basename: str, beetle_position: int, group_img_abs_path: pathlib.Path, indiv_img_abs_path: pathlib.Path, indiv_offset_px: tuple[float, float], individual_id: str, ncc: float, neon_sample_id: str, scientific_name: str)

    ### Instance variables

    `beetle_position: int`
    :

    `group_img_abs_path: pathlib.Path`
    :

    `group_img_basename: str`
    :

    `indiv_img_abs_path: pathlib.Path`
    :

    `indiv_offset_px: tuple[float, float]`
    :

    `individual_id: str`
    :

    `ncc: float`
    :   Normalized cross-correlation score from template matching.

    `neon_sample_id: str`
    :   NEON sample ID.

    `scientific_name: str`
    :   Scientific name (genus species).

    ### Methods

    `to_dict(self) ‑> dict`
    :   Convert annotation to dictionary for JSON serialization.

`Config(hf_root: pathlib.Path = PosixPath('data/beetlepalooza/individual_specimens'), resized_root: pathlib.Path = PosixPath('data/beetlepalooza/group_images_resized'), log_to: pathlib.Path = PosixPath('logs'), dump_to: pathlib.Path = PosixPath('data/beetlepalooza-formatted'), ignore_errors: bool = False, seed: int = 42, sample_rate: int = 20, run_position_correction: bool = True, run_validation_cleanup: bool = True, slurm_acct: str = '', slurm_partition: str = 'parallel', n_hours: float = 2.0, groups_per_job: int = 4)`
:   Config(hf_root: pathlib.Path = PosixPath('data/beetlepalooza/individual_specimens'), resized_root: pathlib.Path = PosixPath('data/beetlepalooza/group_images_resized'), log_to: pathlib.Path = PosixPath('logs'), dump_to: pathlib.Path = PosixPath('data/beetlepalooza-formatted'), ignore_errors: bool = False, seed: int = 42, sample_rate: int = 20, run_position_correction: bool = True, run_validation_cleanup: bool = True, slurm_acct: str = '', slurm_partition: str = 'parallel', n_hours: float = 2.0, groups_per_job: int = 4)

    ### Instance variables

    `dump_to: pathlib.Path`
    :   Where to save formatted data.

    `groups_per_job: int`
    :   Number of group images to process per job.

    `hf_root: pathlib.Path`
    :   Where you dumped data when using download_beetlepalooza.py.

    `ignore_errors: bool`
    :   Skip the user error check and always proceed (equivalent to answering 'yes').

    `log_to: pathlib.Path`
    :   Where to save submitit/slurm logs.

    `n_hours: float`
    :   Number of hours to request for each job.

    `resized_root: pathlib.Path`
    :   I don't know why the huggingface dataset doesn't have the right images. I don't make the rules. I just have to play by them.

    `run_position_correction: bool`
    :   Run row_template_match_rename.py before template matching to correct beetle positions.

    `run_validation_cleanup: bool`
    :   Run validate_beetlepalooza_annotations.py after creating annotations to remove invalid measurements.

    `sample_rate: int`
    :   Save 1 in sample_rate annotations as example images (default: 1 in 20).

    `seed: int`
    :   Random seed for sampling which annotations to save as examples.

    `slurm_acct: str`
    :   Slurm account to use. If empty, uses DebugExecutor.

    `slurm_partition: str`
    :   Slurm partition to use.

`ImageLoadError(group_img_basename: str, message: str, img_path: str)`
:   Error loading an image file.

    ### Ancestors (in MRO)

    * btx.scripts.format_beetlepalooza.WorkerError

    ### Instance variables

    `img_path: str`
    :

`TemplateMatchingError(group_img_basename: str, message: str, beetle_position: int, indiv_img_path: str)`
:   Error during template matching.

    ### Ancestors (in MRO)

    * btx.scripts.format_beetlepalooza.WorkerError

    ### Instance variables

    `beetle_position: int`
    :

    `indiv_img_path: str`
    :

`ValidationError(error_type: Literal['specimen_duplicates', 'missing_files', 'corrupted_files', 'dimension_errors', 'images_without_measurements', 'measurements_without_images'], count: int, details: list[str] = <factory>)`
:   Data validation error with type and count.

    ### Instance variables

    `count: int`
    :

    `details: list[str]`
    :   Optional list of example error details for logging.

    `display_name: str`
    :   Human-readable error type name.

    `error_type: Literal['specimen_duplicates', 'missing_files', 'corrupted_files', 'dimension_errors', 'images_without_measurements', 'measurements_without_images']`
    :

    ### Methods

    `log_summary(self, logger: logging.Logger) ‑> None`
    :   Log a summary of this error.

`WorkerError(group_img_basename: str, message: str)`
:   Base class for worker errors with context.

    ### Descendants

    * btx.scripts.format_beetlepalooza.ImageLoadError
    * btx.scripts.format_beetlepalooza.TemplateMatchingError

    ### Instance variables

    `group_img_basename: str`
    :

    `message: str`
    :