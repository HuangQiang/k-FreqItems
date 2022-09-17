#include "lsh.cuh"

namespace clustering {

// -----------------------------------------------------------------------------
int uni_hash[100] = { 7,73,97,751,397,251,769,431,139,167,997,367,421,569,821,
    733,163,947,739,941,751,41,131,293,709,233,181,223,263,929,151,541,733,89,
    907,373,101,281,149,859,647,911,59,653,257,907,947,197,953,647,113,19,233,
    313,599,929,991,743,421,907,317,571,383,401,797,971,719,397,739,787,239,
    439,967,149,587,73,971,23,31,223,419,431,379,167,479,677,269,257,571,67,
    317,947,79,271,919,929,251,173,617,29, };

// -----------------------------------------------------------------------------
int fixedcoeffA[1000] = { 6517155,4843265,456393,696809,3033337,737017,4568968,
    6240157,6816902,166900,8267017,8054957,3324705,3747952,8377086,2952217,
    3672878,7327488,2742231,2355981,3664319,4042611,2957148,5139656,4396424,
    7911966,2933883,7713940,6489839,4149293,1517455,3006993,8992559,1973849,
    3703803,2025895,2710866,8272771,8266052,9527768,8439671,6533068,7582724,
    1764374,281018,5959809,4716592,3953896,3287295,7458823,6309878,6951614,
    1501433,9267027,2091268,5897857,7178991,5025152,3611796,3668829,9174445,
    5129252,6675823,8167002,7103101,379624,192895,9813967,8652395,8458948,
    9341734,7092065,4992014,6924457,8856439,5273033,2884264,3573029,9226930,
    6171559,1031851,5536806,3123172,2533284,4803832,5214440,8431142,1982822,
    239590,2042936,5651651,9414036,7172189,2327472,7581036,4275288,2707096,
    7773932,4089254,1359490,6232879,3430987,8451555,1224891,355442,7307993,
    6497924,3239706,881021,5724852,9411266,1912872,1261657,2534436,4446157,
    6065489,7748877,2877297,8048312,7988467,4920234,3699961,7402501,2092421,
    6027434,4983536,6367709,8734531,2757467,456962,94019,8990346,3887949,
    8545575,215235,4243391,5853566,6713160,7483098,6734587,2438011,6894362,
    8647460,3699668,9428798,3093615,9765158,7177673,5970912,7813468,5166139,
    891145,1513428,2568639,2983566,7540862,7552176,9351276,6275392,309641,
    9808238,6369412,9299988,3696185,4914985,9515223,7939577,768550,6228382,
    5422673,7503137,8666393,2317033,6150596,2366060,1745830,9244211,2131216,
    8923504,5215122,9944685,4089641,6106267,1458111,6658280,9089833,8998974,
    4210455,8441107,5274364,4520096,8249344,1643775,3820082,1945528,6558760,
    3335304,9885105,7327311,9563687,5307776,4830447,8230078,7624810,981041,
    596137,9370640,225250,2727353,8294142,5440372,2672036,2383782,1546637,
    4130148,9042063,636469,3129120,3252516,9077577,8403485,7772613,7326919,
    47258,1592694,9272447,6606019,4927999,9157551,3933328,4491684,4465325,
    8763776,2721761,2090134,9744817,3317898,1460773,9970068,6045252,9754915,
    5410438,8717289,2138696,6957076,2847435,1180757,7593546,5976556,4433274,
    6671121,4380039,2205886,3998039,4427298,3798581,3270484,1033316,8726580,
    2428033,4966645,3218263,6893359,3730419,5940025,8983493,3475235,9257924,
    444264,3445301,5303175,199178,8855740,4020462,2337875,5812814,6867897,
    3518633,3406358,2844452,7951907,77478,7224491,157792,4075517,1651788,
    3956373,7346002,2685104,2682952,9774036,7651750,5901215,6667393,1382167,
    1841239,5650885,4857402,1099162,6095150,6621590,5417718,5397056,4130962,
    4297141,2733953,2387360,8773507,4167849,1718983,1533126,5517738,9351848,
    6482261,6071894,3060709,5361062,9389566,8911599,340876,7131024,1151684,
    7542142,334077,4599347,4280534,3309643,2333109,6801555,5030420,3289488,
    5566932,6569555,9408667,8477025,8689893,2824371,5126431,9766766,2076503,
    7513567,498585,6155307,3484700,1647063,5830310,6606247,3587099,6353738,
    8656955,2745308,438810,7188423,5936191,1025747,1321302,4369730,9362522,
    4651580,3481898,1362340,1235192,709447,1144388,7818056,1893947,7148188,
    6133041,7897996,1291800,81924,7463055,4432987,4976690,9672797,5383208,
    7770794,9003723,1440690,5543425,9949452,9822109,9770126,1434250,8024381,
    6591864,5339149,2557958,880495,8856701,5993234,1588428,858308,2244267,
    740260,1557744,5225061,7787350,7335001,2663309,1947648,3465686,3310091,
    7058932,3558290,4743876,155541,3315550,6924454,3312847,3347507,8354425,
    6250441,3567755,5363374,8944824,5486235,4597818,8434152,2986909,7572382,
    1543964,4063088,8741772,419330,8652255,6925208,2339404,3148453,6761091,
    2138221,5822848,8116422,6420780,2077415,8543014,7138474,7377720,4946492,
    2189103,8174109,9132657,7848036,8274965,7379219,9508314,741969,7141895,
    5751826,5408964,2970363,7479037,4038955,1310397,976246,4351267,4340856,
    4249672,4955985,3185379,7560692,1852643,8024853,7161051,2557212,1605919,
    6521991,5636411,3941710,9546392,446307,4068310,7381581,6976053,842617,
    4672768,2855124,6705968,5519949,4656655,6437032,3604735,5347527,3519968,
    6187953,4207191,6964017,1421421,6593482,6683163,8558905,5839407,7881342,
    2294005,6412393,1478450,3213231,5808448,4324330,7849994,2394775,7146540,
    332685,3704389,6965433,4362265,1743482,6938365,2747047,3026113,6183028,
    2008864,9587747,7957016,7145726,4424359,7864194,8899292,138859,5123045,
    2991232,2628628,9686982,4359536,8799362,1593467,1890211,4378647,1419641,
    5390034,8142671,8105867,1385291,4203743,9608838,8628440,8812097,7334782,
    824060,933105,9079967,4605809,6128900,2448135,1320117,6113907,7652561,
    5469653,1224931,561999,9000182,3433837,447505,6311355,2462937,894985,
    9771030,2513080,4485248,3448470,2799382,5268153,3782149,9489793,411199,
    5840848,9242230,7223658,43303,961299,779471,4049142,7357077,6450430,
    1523435,7647690,9461276,4774759,3653647,8250503,4316343,3476226,3366034,
    3248307,4234895,5915576,6065889,5564865,6526938,5265290,1914194,2373188,
    5828821,4618893,1412498,3547312,6960746,9287412,7723420,3736116,6745323,
    6972679,9616407,6234176,4552838,4193716,4412444,7842500,3714850,9723731,
    5330788,856187,294580,2618312,2679315,9976314,6006117,2235849,9486536,
    1765868,9969490,506405,2526487,4623198,810297,1368683,7617971,8432617,
    2368467,5930550,7405356,4472419,2855854,9033996,5178293,6381763,6759446,
    9529715,5237072,5527268,8499840,2188453,9120819,4030360,1671999,6537067,
    5867771,2488826,7717753,1066014,2322684,8043827,4335943,8368061,1726454,
    5620384,791278,6344103,8429907,1401382,335371,9179259,8947933,4926766,
    3190905,9561731,3440555,4595718,5179098,7349513,6859106,7131845,778733,
    7562480,3850410,9310543,6902310,388922,157885,2515928,1308466,8862379,
    9683215,2525519,6421874,1767864,2844927,931363,2375655,2832297,7457705,
    411844,632641,210255,6116452,4139315,4803841,2173937,6459789,9349367,
    4574486,6162643,16938,1043071,5587848,8618656,6018207,7405566,9831264,
    7441372,342003,3372631,4582379,9632898,246023,3923831,99495,2239921,
    7180478,2509024,7158359,5813878,158178,9827412,835263,5937155,8849469,
    6323109,4856354,4474015,4938270,9922885,8484318,5154195,5445797,8433121,
    8790898,5780720,7466441,8938516,2876914,9478969,4185457,5158230,1758082,
    4144187,8030089,1734115,6229877,2179201,4551528,4279829,4859319,2291500,
    5668681,627462,6105687,271977,5908507,9332953,5062450,4362939,7544210,
    9606945,2985363,5449761,2398466,879315,9790138,9583543,4400831,4004412,
    317434,797659,9706594,3892528,3607749,7861582,4352520,4926045,9254672,
    4874772,105394,8910921,8924901,5354350,8705468,3832248,3635150,5248841,
    4760517,8110786,7396835,9520224,3440894,9087236,4822074,4348702,9716093,
    4823000,1754086,1966725,9536111,9482454,428877,6513003,3872663,3557659,
    596157,3018851,7524150,6732615,6068093,7273691,4340939,4425178,2274039,
    6554576,2190874,4981409,768323,7809143,6118201,5910482,4444683,751407,
    1035419,4315056,8565338,2887695,3685436,9001300,3704775,7769623,1592904,
    8515399,4245443,7002212,7691343,7077999,1842026,4104857,5313428,9980955,
    8659,6785042,181523,9653567,1342217,9781946,174221,5296348,4621885,
    4783305,5211253,4449252,9690710,9930722,6974021,8882902,2780326,676772,
    1454285,6848146,7753528,5731622,4927434,2347411,3326527,1862545,8992020,
    5591057,4865904,3686866,5433456,8822603,8237711,1175630,9829714,2488496,
    4653865,5510989,9248085,9130616,4989116,1112391,7304956,2663864,2225227,
    4010783,8920497,5872463,2332760,6850201,5071866,9745714,7558812,9307029,
    3115102,9381508,9430805,2085352,7197346,8906619,1099944,7081253,5750786,
    7178646,1808,246017,3367542,4799404,5248433,7733250,2544076,6467751,
    7608656,8710639,3302316,4228573,534658,4422699,4263542,4368988,5630191,
    9339022,4606735,9924087,8244856,8218700,4772094,2763811,3239552,7186752,
    9091706,944127,2725151,5911806,6968171,1254040,4895798,1596094,4731486,
    9357698,9177516,3765272,2262849,6330360,8003531,2044268,4932435,6721599,
    8355220,4646276,2914198,7921751,6493377,7312528,1038024,4201202,4246623,
    4165149,4495555,2011871,4012619,9594321,7044532,6602638,8949325,723142,
    7149229,9673518,8162736,4867467,3656270,9416299,7035143,4412364,4076956,
    4029783,480372,7034628,9940023,2924655,4886179,752869,2292688,5678013,
    7412430,9656184,9614148,8971592,2653930,8179987,9808188,4820460,5833872,
    7813463,5492352,4663241,};

// -----------------------------------------------------------------------------
int fixedcoeffB[1000] = { 4912045,1121710,6068228,4062045,6585752,6977557,
    7609116,2860722,7040660,4123844,2006871,6934457,1107623,6983831,9381675,
    3138533,5024471,7871574,2539842,1269105,9140341,9382344,1533366,6759648,
    4408022,531611,511108,7122767,9375412,3176618,1217891,4287455,4298328,
    7286120,8349501,884079,4263675,5958615,3744802,1304334,82457,5751674,
    8238791,1190081,2735503,7620464,4328614,7759974,5492036,6868457,9029079,
    4632376,6250800,562444,1392022,658820,1094056,1903130,7781588,469466,
    5079749,8999479,4756922,9378077,6285597,3106421,262155,549271,9065037,
    4006957,1853606,9147495,9758631,92396,337574,2494132,7712861,4666189,
    254105,3204896,1534644,9283185,7837272,7785444,9845630,9229294,8444265,
    939684,1132423,6225851,1409151,6212172,5225329,6166074,5590248,1510924,
    9272496,5852403,2060196,8337531,9859360,3913802,7485025,9617990,4006199,
    7822599,2112121,1719058,2488786,2366226,4923954,4023431,1649410,2761224,
    1808874,1495038,1990517,253137,2434723,3122940,6478988,3843875,9335113,
    1704315,9947,4925359,3215240,9282444,777761,5275437,7619974,637120,9189240,
    5104997,255108,3195437,2927595,2367229,4914495,5416381,4733455,9838449,
    9439813,6382866,2599672,1248685,7877904,4590189,1501822,312626,7713130,
    7980811,4156501,7048241,9685127,4166449,1973599,2900366,3448891,2751360,
    8175803,1068864,3388481,7365041,6173861,3643589,560476,9101457,6010818,
    5474972,4517837,744272,5313419,3957648,7127138,7913092,5206333,5005041,
    2503280,6708156,5317667,216408,4688966,9474169,7264650,4374092,3640617,
    9238250,7274458,7089509,1989609,5450259,8158373,5378090,2815299,4332233,
    9021679,3375776,3433688,5032495,8850748,7951525,5776768,4164166,1909172,
    2903904,2077256,7115506,7908946,4580536,3823661,3226612,4796945,8512627,
    2700780,2061594,2886717,6341397,1299842,161174,3430904,3289451,5611433,
    1589275,8667541,8426733,5921509,7689218,1802507,9355197,2721712,653253,
    7306721,8498481,4817419,9215893,1402384,6894676,6331397,9311330,1475211,
    155056,2537940,6272156,8667684,5238721,8333750,1554400,1580116,9633592,
    1715574,5011020,2923042,7327008,6600296,1590581,5753739,2521803,9279800,
    7556246,1876999,2001511,8209500,9183720,499991,3026918,8399612,1902375,
    9921594,4731008,1213703,1396803,4886065,3751644,7668960,3553748,8990365,
    6002709,5108148,570480,5636300,6823722,5581500,8559342,4150728,2181795,
    149922,9904467,4703598,9429723,7460712,6580598,1431232,5670210,5764317,
    1931223,8697128,4163927,3833599,8618721,886945,6696083,433174,9582758,
    3610496,7942945,4467314,815695,4674157,2570543,1256584,6200637,7003284,
    2368930,4474118,2693484,8478426,758118,3520528,3208740,3074829,9518383,
    9925190,1709991,2053060,6132473,9798271,2661491,8815370,2873448,8091129,
    8650550,4956498,5481154,9749543,5608048,9841577,366513,5460507,4366113,
    6357037,823209,2831694,2956862,8515119,4936550,1397241,3175802,2995766,
    4830762,7346847,5569678,5565240,6955189,8012694,8539313,1819887,525037,
    2847221,6298642,9312207,7968587,4822291,7063185,9801342,563364,2332756,
    4336611,2069231,8923743,2613101,4451654,6825577,9084566,8458587,1035137,
    6618400,8480240,148110,9258746,7437371,7097639,5955150,7668122,2726049,
    8106305,4054919,9772136,7534647,553292,4239909,3764027,592151,2298998,
    7787691,689711,1379852,4995187,9221600,5359382,9255172,7286797,6023643,
    4190585,5915633,9551524,6332304,1013849,5559817,7664714,4870590,8572568,
    6141775,3585442,6231063,7311022,2410509,8796325,1656750,1992441,7730785,
    7727923,4859514,987542,2054619,3766837,7006679,3175649,5907159,8511826,
    5705914,7624456,420665,6550573,9407689,9649797,86978,1201562,6119491,
    5365821,841357,5099317,1472061,996488,5036056,9517440,5331161,8858391,
    3562675,7829240,3951978,108836,4856756,8985446,3499500,9150355,3483924,
    2188891,2524636,2335204,2693693,8302660,8994538,3776679,2570014,5527575,
    9084955,560946,6946808,4917956,3897658,4787162,8317981,2337565,161432,
    8403309,6525411,879977,3867019,8077161,6262573,2959023,9241146,9289236,
    3066360,6634259,8275501,4345598,8397473,7436099,1355931,1414028,2545375,
    3401290,6950659,2849425,5083185,8560758,9328416,512282,2057185,9358792,
    8567960,6926167,387552,5524337,201672,9624823,9232369,9159439,4504563,
    9396259,7817311,3912077,5807009,9540501,1570899,8196879,2603681,7104291,
    9485310,4734713,9566993,2434028,7385649,4982389,3602330,1477958,2335662,
    8565367,9814393,6006865,7005158,1524149,9326787,2600069,1257067,8499078,
    1694318,9623421,2526571,970177,8984455,7548541,7838169,9462788,4219773,
    9037851,9673395,833768,9190439,7812278,768619,1271565,7244459,1542903,
    5500794,5899930,4961214,5805365,9562377,451943,737805,5007080,51792,
    9411380,9274684,9689735,3535012,2506239,1856284,550501,6562221,1842325,
    9898900,4322817,2915842,3243424,4264551,4064846,4091350,3558572,699883,
    3409337,4209606,5014365,9964717,3422965,2015295,8050372,2912979,1375469,
    7147947,9482467,2869396,5728840,7023538,326779,2535718,1958314,9651691,
    2811212,2537544,6143344,1499466,6467032,6566904,241264,2461393,5127347,
    3271043,2291533,143598,8018000,6354427,9424638,6948992,5622524,8470025,
    4039374,5959583,4381934,8348905,6241336,4346928,6141085,2224869,7912550,
    1050929,5047782,5906863,3411473,6708268,2342390,799863,5147782,6888168,
    9285932,4152913,6047369,6895950,6117810,621070,1489286,9960909,9077354,
    1009486,8723625,1522822,2896045,7447776,9082129,7573542,9573015,3895328,
    226293,7808837,9912097,374463,101905,7938448,6203142,1668084,6665686,
    5992779,5513589,7605557,7776498,2963778,6905552,2969896,7650384,3475947,
    9208639,8963862,4185780,8489801,5850242,5315687,3348790,2077658,4749021,
    3458096,4862313,9875031,3280980,8899708,3623659,8493843,6226064,2142840,
    4360014,800765,790525,2726183,7424917,3001693,4487055,4699456,2881663,
    1117231,7419413,7094006,2392592,3875368,3045553,9030499,6650847,7615821,
    5947715,1580686,5145843,5546568,1042168,8982264,4820509,86473,116432,
    3282992,2768326,1127679,3176565,3219443,9989441,7268783,1177266,9309786,
    3431848,5956006,5102377,8397845,5037764,2162806,5664795,7663219,2604156,
    4246998,8949075,5608132,8301703,4875671,1726383,5802077,9041810,9632245,
    6696351,4067072,6514406,6888310,1675674,612321,3342425,60549,2325759,
    1230016,8545594,106430,291143,5430206,7666588,3477139,3092539,6097053,
    3774087,1786145,7536890,9729590,7257784,3170815,5458818,211005,506517,
    4276388,6225940,130611,4769044,8255467,1151959,7468527,7337882,375805,
    1653137,5754087,9895532,8494464,3325731,9358762,9022901,4629906,3341743,
    9141369,8368266,5972368,1749354,4446269,2695565,3962007,4927468,3259622,
    8640994,177349,459553,5798972,3171260,4731294,5189393,4623789,1269497,
    3077882,4037051,2106086,2346224,1081317,5784185,2096690,6095024,6391903,
    1810149,6037029,2619751,1494412,4640985,1366801,6714357,8929333,1968833,
    7484022,7103511,1588746,2906277,7312873,5959474,5555876,5895649,2751031,
    2361227,6324542,363643,2912694,2136025,7829406,2877445,8419913,3878766,
    2868650,3694094,5786342,3886298,7423190,2935121,8297344,4419663,2298559,
    2313227,1861304,7053278,8554109,5004137,9911677,6982679,5667944,2961849,
    330339,2796501,6630091,7927749,1027969,9549318,7130716,8537779,6311796,
    8682729,5521779,1839924,2569768,8213782,9499375,9691995,8023775,7583241,
    8220117,9581298,7858400,2574612,993161,3981135,6703448,2170859,6350729,
    7658682,2833385,7158207,7048092,2162473,9338696,9725731,5256703,6806897,
    8437644,8104217,493899,6735760,4749215,9422572,2917401,20440,2953716,
    8083545,4028847,5073673,9991730,926354,4106432,8363534,7114758,1974880,
    8553102,4806002,7617257,4402260,1309826,6285443,1601345,4265349,4615005,
    8997733,4138425,9855167,7657337,788931,4686451,2380750,1474449,6541868,
    1415325,9626363,5366826,6988692,280794,2598361,593061,9502532,1520180,
    2976341,7576398,3949610,6529082,5502400,3517111,7095236,2988394,7657585,
    2633405,5323390,5900316,7809996,9517845,1224862,6670226,215540,7966474,
    6509471,495509,2499086,4056138,8945054,6315920,5399752,1248914,1863522,
    8644892,3642606,6986045,9200760,9068669,7902893,3581448,1530697,6424234,
    8089575,3624103,7524904,3637265,8889347,3647797,2035408,426386,455424,
    545480,1068700,9072483,6649290,3383955,6730886,2209769,288560,1855088,
    5773895,738118,378531,};

// -----------------------------------------------------------------------------
template<class DType>
__global__ void sign_minhash(       // calc minhash value
    int   batch,                        // number of data points/buckets
    int   sign_size,                    // sign_size = l * k
    int   prime,                        // prime number for dimension
    const int   *d_coeA,                // permutation hash coefficients
    const int   *d_coeB,                // permutation hash coefficients
    const DType *d_set,                 // data/bucket set
    const u64   *d_pos,                 // data/bucket position
    int   *d_sig)                       // hash values of data/bucket (return)
{
    // d_sig has <batch> chunks, and each chunk has <sign_size> hash values
    u64 tid = (u64) blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < (u64) batch * sign_size) {
        int   did = tid / sign_size;            // get data/bucket id
        int   num = get_length(did, d_pos);     // get data/bucket size
        const DType *data = d_set + d_pos[did]; // get data/bucket
        
        int sid  = tid % sign_size; // signature id
        int coeA = d_coeA[sid], coeB = d_coeB[sid];
        
        // scan and find the minimum order as hash value of this data/bucket
        int val = ((u64) data[0] * coeA + coeB) % prime;
        for (int i = 1; i < num; ++i) {
            int tmp = ((u64) data[i] * coeA + coeB) % prime;
            if (tmp < val) val = tmp;
        }
        d_sig[tid] = val;
    }
}

// -----------------------------------------------------------------------------
__global__ void sign_concat(        // universal hash to get k-concatenation
    int   batch,                        // number of batch data
    int   l,                            // number of hash values
    int   k,                            // k-concatenation
    int   prime,                        // prime number for data/buckets
    const int *d_sig,                   // signatures
    const int *d_hash,                  // universal hash function
    int   *d_val)                       // hash values (return)
{
    // d_val has <batch> chunks, and each chunk has <l> signatures
    u64 tid = (u64) blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < (u64) batch * l) {
        const int *sig = d_sig + (u64) tid*k; // get signatures
        
        // universal hashing for k-concatenation
        u64 val = 0UL;
        for (int i = 0; i < k; ++i) {
            // val & MAX_UINT32: low-32-bit, val >> 32: high-32-bit
            val += (u64) d_hash[i] * (u64) sig[i];
            val = (val & MAX_UINT32) + 5 * (val >> 32);
            if (val > UINT32_PRIME) val -= UINT32_PRIME;
        };
        d_val[tid] = val % prime;
    }
}

// -----------------------------------------------------------------------------
__global__ void combine(            // combine two k-concatenations together
    int   batch,                        // number of batch data
    int   m,                            // number of hash tables
    int   l,                            // number of hash values
    int   prime,                        // prime number
    const int *d_val,                   // hash values
    int   *d_res)                       // hash results (return)
{
    // d_res has <m> chunks, and each chunk has <batch> signatures
    u64 tid = (u64) blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < (u64) batch) {
        const int *val = d_val + (u64) tid*l; // get hash values
        
        int cnt = 0;
        for (int i = 0; i < l; ++i) {
            for (int j = i+1; j < l; ++j) {
                u64 pos = (u64) cnt * batch + tid; // get position
                d_res[pos] = ((u64) val[i] + val[j]) % prime;
                if (++cnt >= m) break;
            }
            if (cnt >= m) break;
        }
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void minhash(                       // minwise hashing (pair-wise)
    int   rank,                         // MPI rank
    int   n,                            // number of data points/buckets
    int   n_prime,                      // prime number for data/buckets
    int   d_prime,                      // prime number for dimension
    int   m,                            // #hash tables
    int   h,                            // #concatenated hash func
    const DType *dataset,               // data/bucket set
    const u64   *datapos,               // data/bucket position
    int   *hash_results)                // hash results (return)
{
    cudaSetDevice(DEVICE_LOCAL_RANK);
    
    assert(h % 2 == 0);
    int k = h / 2;
    int l = (int) ceil(sqrt(2.0f*m)) + 1;
    // only have 1,000 static coeff values and 100 uni_hash values
    if (l*k > MAX_NUM_HASH || k > 100) exit(1); 
    
    // init hash parameters 
    int *d_coeA; cudaMalloc((void**)&d_coeA, sizeof(int)*l*k);
    int *d_coeB; cudaMalloc((void**)&d_coeB, sizeof(int)*l*k);
    int *d_hash; cudaMalloc((void**)&d_hash, sizeof(int)*k);
    
    cudaMemcpy(d_coeA, fixedcoeffA, sizeof(int)*l*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_coeB, fixedcoeffB, sizeof(int)*l*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash, uni_hash,    sizeof(int)*k,   cudaMemcpyHostToDevice);
    
    // mem_avail = totoal_size - memory(d_coeA + d_coeB + d_hash)
    u64 mem_avail = GPU_MEMORY_LIMIT - sizeof(int)*(l*k*2+k);
    u64 mem_usage = 0UL, n_set = 0UL;
    int n_pos = 0, batch = 0, start = 0;
    
    for (int i = 0; i <= n; ++i) {
        // ---------------------------------------------------------------------
        //  calculate memory usage requirement if adding one more data
        // ---------------------------------------------------------------------
        if (i < n) {
            // d_set + d_pos + d_sig + d_val + d_res
            n_set = datapos[i+1] - datapos[start];
            n_pos = batch + 2;
            mem_usage = sizeof(DType)*n_set + sizeof(u64)*n_pos + 
                sizeof(int)*((u64) l*k+l+m)*(batch+1);
        }
        // ---------------------------------------------------------------------
        //  parallel minhash for bucket if over mem_avail or end
        // ---------------------------------------------------------------------
        if (mem_usage > mem_avail || i == n) {
            // d_set + d_pos + d_sig + d_val + d_res
            n_set = datapos[i] - datapos[start];
            n_pos = batch + 1;
            mem_usage = sizeof(DType)*n_set + sizeof(u64)*n_pos + 
                sizeof(int)*((u64) l*k+l+m)*batch;
#ifdef DEBUG_INFO
            printf("Rank #%d: n=%d, i=%d, batch=%d, mem_usage=%lu, mem_avail=%lu\n", 
                rank, n, i, batch, mem_usage, mem_avail);
#endif
            // cuda allocation and memory copy from CPU to GPU
            const DType *h_set = dataset + datapos[start]; // dataset at host
            int *h_res = new int[(u64) batch*m];           // results at host
            u64 *h_pos = new u64[n_pos];              // data postion at host
            copy_pos(n_pos, datapos + start, h_pos);

            // cuda allocation: dataset, data position, signatures, hash results
            DType *d_set; cudaMalloc((void**) &d_set, sizeof(DType)*n_set);
            u64   *d_pos; cudaMalloc((void**) &d_pos, sizeof(u64)*n_pos);
            int   *d_sig; cudaMalloc((void**) &d_sig, sizeof(int)*batch*l*k);
            int   *d_val; cudaMalloc((void**) &d_val, sizeof(int)*batch*l);
            int   *d_res; cudaMalloc((void**) &d_res, sizeof(int)*batch*m);
            
            cudaMemcpy(d_set, h_set, sizeof(DType)*n_set, cudaMemcpyHostToDevice);
            cudaMemcpy(d_pos, h_pos, sizeof(u64)*n_pos,   cudaMemcpyHostToDevice);
            
            // calc minhash values for batch data by GPUs
            int block = BLOCK_SIZE;
            int grid  = ((u64) batch*l*k + block-1) / block;
            sign_minhash<DType><<<grid, block>>>(batch, l*k, d_prime, d_coeA, 
                d_coeB, d_set, d_pos, d_sig);
            
            // calc the k-concatenation by GPUs
            grid = ((u64) batch*l + block-1) / block;
            sign_concat<<<grid, block>>>(batch, l, k, n_prime, d_sig, d_hash, d_val);
            
            // calc the final m hash values by GPUs
            grid = ((u64) batch + block-1) / block;
            combine<<<grid, block>>>(batch, m, l, n_prime, d_val, d_res);
            
            cudaMemcpy(h_res, d_res, sizeof(int)*batch*m, cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize();
            
            for (int j = 0; j < m; ++j) {
                const int *res = h_res + (u64) j*batch;
                int *hash_res  = hash_results + (u64) j*n+start;
                std::copy(res, res + batch, hash_res);
            }
            // release space
            cudaFree(d_set); cudaFree(d_pos);
            cudaFree(d_res); cudaFree(d_val); cudaFree(d_sig);
            delete[] h_pos;  delete[] h_res;
            
            // update local parameters for next minhash
            start += batch; batch = 0;
        }
        if (i < n) ++batch;
    }
    // release space
    assert(start == n);
    cudaFree(d_hash); cudaFree(d_coeA); cudaFree(d_coeB);
}

// -----------------------------------------------------------------------------
template void minhash(              // minwise hashing (pair-wise)
    int   rank,                         // MPI rank
    int   n,                            // number of data points/buckets
    int   n_prime,                      // prime number for data/buckets
    int   d_prime,                      // prime number for dimension
    int   m,                            // #hash tables
    int   h,                            // #concatenated hash func
    const u08 *dataset,                 // data/bucket set
    const u64 *datapos,                 // data/bucket position
    int   *hash_results);               // hash results (return)
    
// -----------------------------------------------------------------------------
template void minhash(              // minwise hashing (pair-wise)
    int   rank,                         // MPI rank
    int   n,                            // number of data points/buckets
    int   n_prime,                      // prime number for data/buckets
    int   d_prime,                      // prime number for dimension
    int   m,                            // #hash tables
    int   h,                            // #concatenated hash func
    const u16 *dataset,                 // data/bucket set
    const u64 *datapos,                 // data/bucket position
    int   *hash_results);               // hash results (return)
    
// -----------------------------------------------------------------------------
template void minhash(              // minwise hashing (pair-wise)
    int   rank,                         // MPI rank
    int   n,                            // number of data points/buckets
    int   n_prime,                      // prime number for data/buckets
    int   d_prime,                      // prime number for dimension
    int   m,                            // #hash tables
    int   h,                            // #concatenated hash func
    const int *dataset,                 // data/bucket set
    const u64 *datapos,                 // data/bucket position
    int   *hash_results);               // hash results (return)
    
// -----------------------------------------------------------------------------
template void minhash(              // minwise hashing (pair-wise)
    int   rank,                         // MPI rank
    int   n,                            // number of data points/buckets
    int   n_prime,                      // prime number for data/buckets
    int   d_prime,                      // prime number for dimension
    int   m,                            // #hash tables
    int   h,                            // #concatenated hash func
    const f32 *dataset,                 // data/bucket set
    const u64 *datapos,                 // data/bucket position
    int   *hash_results);               // hash results (return)

// -----------------------------------------------------------------------------
template<class DType>
__global__ void signature(          // calc signature
    int   batch,                        // number of batch data
    int   sign_size,                    // sign_size = l*k,
    int   d,                            // data dimension
    float w,                            // bucket width
    const DType *d_data,                // batch data
    const float *d_proj,                // random projection
    const float *d_shift,               // random shift
    int   *d_sig)                       // signatures (return)
{
    // d_sig has <batch> chunks, and each chunk has <sign_size> signatures
    u64 tid = (u64) blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < (u64) batch * sign_size) {
        int   did = tid / sign_size; // get data ID
        int   pid = tid % sign_size; // get proj ID
        
        const DType *data = d_data + (u64) did*d;
        const float *proj = d_proj + pid*d;
        
        float shift = d_shift[pid];
        float val = thrust::inner_product(thrust::device, data, data+d, proj, 0.0f);
        d_sig[tid] = (int) floor((val + shift) / w);
    }
}

// -----------------------------------------------------------------------------
__global__ void concatenation(      // universal hash to get k-concatenation
    int   batch,                        // number of batch data
    int   l,                            // number of hash values
    int   k,                            // k-concatenation
    int   prime,                        // prime number
    const int *d_sig,                   // signatures
    const int *d_hash,                  // universal hash function
    int   *d_val)                       // hash values (return)
{
    // d_val has <batch> chunks, and each chunk has <l> signatures
    u64 tid = (u64) blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < (u64) batch * l) {
        const int *sig = d_sig + (u64) tid*k; // get signature
        
        // universal hashing for k-concatenation
        u64 val = 0UL;
        for (int i = 0; i < k; ++i) {
            // val & MAX_UINT32: low-32-bit, val >> 32: high-32-bit
            // NOTE: sig[i] can be negative !!!
            val += (u64) d_hash[i] * (u32) sig[i];
            val = (val & MAX_UINT32) + 5 * (val >> 32);
            if (val > UINT32_PRIME) val -= UINT32_PRIME;
        }
        d_val[tid] = val % prime;
    }
}

// -----------------------------------------------------------------------------
template<class DType>
void e2lsh(                         // calc hash results using e2lsh
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   prime,                        // prime number
    int   m,                            // number of hash tables
    int   h,                            // number of concat hash functions
    float w,                            // bucket width
    const float *proj,                  // random projection, m*h*d
    const float *shift,                 // random shift, m*h
    const DType *dataset,               // data set
    int   *hash_results)                // hash results (return)
{
    cudaSetDevice(DEVICE_LOCAL_RANK);
    
    assert(h % 2 == 0);
    int k = h / 2;
    int l = (int) ceil(sqrt(2.0f*m)) + 1;
    // only have 1,000 static coeff values and 100 uni_hash values
    if (l*k > MAX_NUM_HASH || k > 100) exit(1);
    
    // init hash parameters
    float *d_proj;  cudaMalloc((void**)&d_proj,  sizeof(float)*l*k*d);
    float *d_shift; cudaMalloc((void**)&d_shift, sizeof(float)*l*k);
    int   *d_hash;  cudaMalloc((void**)&d_hash,  sizeof(int)*k);
    
    cudaMemcpy(d_proj,  proj, sizeof(float)*l*k*d, cudaMemcpyHostToDevice);
    cudaMemcpy(d_shift, shift,  sizeof(float)*l*k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hash,  uni_hash,   sizeof(int)*k, cudaMemcpyHostToDevice);
    
    // mem_avail = total_size - memory(d_proj + d_shift + d_uni_hash)
    // batch = mem_avail / memory(d_data + d_sig + d_val + d_res)
    u64 mem_avail = GPU_MEMORY_LIMIT - (sizeof(float)*l*k*(d+1) + sizeof(int)*k);
    u64 mem_one_cost = sizeof(DType)*d + sizeof(int)*(l*k+l+m);
    u64 batch = mem_avail / mem_one_cost;
    if (batch > n) batch = n;
    
    // init parameters for batch data
    int   *h_res = new int[(u64) batch*m];
    DType *d_data; cudaMalloc((void**) &d_data, sizeof(DType)*batch*d);
    int   *d_sig;  cudaMalloc((void**) &d_sig,  sizeof(int)*batch*l*k);
    int   *d_val;  cudaMalloc((void**) &d_val,  sizeof(int)*batch*l);
    int   *d_res;  cudaMalloc((void**) &d_res,  sizeof(int)*batch*m);

    // compute random projections
    for (int i = 0; i < n; i += batch) {
        if (i+batch > n) batch = n-i;
#ifdef DEBUG_INFO
        printf("Rank #%d: n=%d, i=%d, batch=%d\n", rank, n, i, batch);
#endif
        // copy data from CPU memory to GPU memory
        const DType *h_data = dataset + (u64) i*d;
        cudaMemcpy(d_data, h_data, sizeof(DType)*batch*d, cudaMemcpyHostToDevice);
        
        // calc h_{a,b}(o)=floor((<a,o>+b)/w) by GPUs
        int block = BLOCK_SIZE;
        int grid  = ((u64) batch*l*k + block-1) / block;
        signature<DType><<<grid, block>>>(batch, l*k, d, w, d_data, d_proj,
            d_shift, d_sig);
        
        // calc the k-concatenation by GPUs
        grid = ((u64) batch*l + block-1) / block;
        concatenation<<<grid, block>>>(batch, l, k, prime, d_sig, d_hash, d_val);
        
        // calc the final m hash values by GPUs
        grid = ((u64) batch + block-1) / block;
        combine<<<grid, block>>>(batch, m, l, prime, d_val, d_res);
        
        // update results
        cudaMemcpy(h_res, d_res, sizeof(int)*batch*m, cudaMemcpyDeviceToHost);
        // cudaDeviceSynchronize();
        
        for (int j = 0; j < m; ++j) {
            const int *res = h_res + (u64) j*batch;
            int *hash_res  = hash_results + (u64) j*n+i;
            std::copy(res, res + batch, hash_res);
        }
    }
    // free memory
    cudaFree(d_proj); cudaFree(d_shift); cudaFree(d_hash);
    cudaFree(d_data); cudaFree(d_sig); cudaFree(d_val); cudaFree(d_res);
    delete[] h_res;
}

// -----------------------------------------------------------------------------
template void e2lsh(                // calc hash results using e2lsh
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   prime,                        // prime number
    int   m,                            // number of hash tables
    int   h,                            // number of concat hash functions
    float w,                            // bucket width
    const float *proj,                  // random projection, m*h*d
    const float *shift,                 // random shift, m*h
    const u08   *dataset,               // data set
    int   *hash_results);               // hash results (return)

// -----------------------------------------------------------------------------
template void e2lsh(                // calc hash results using e2lsh
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   prime,                        // prime number
    int   m,                            // number of hash tables
    int   h,                            // number of concat hash functions
    float w,                            // bucket width
    const float *proj,                  // random projection, m*h*d
    const float *shift,                 // random shift, m*h
    const u16   *dataset,               // data set
    int   *hash_results);               // hash results (return)

// -----------------------------------------------------------------------------
template void e2lsh(                // calc hash results using e2lsh
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   prime,                        // prime number
    int   m,                            // number of hash tables
    int   h,                            // number of concat hash functions
    float w,                            // bucket width
    const float *proj,                  // random projection, m*h*d
    const float *shift,                 // random shift, m*h
    const int   *dataset,               // data set
    int   *hash_results);               // hash results (return)

// -----------------------------------------------------------------------------
template void e2lsh(                // calc hash results using e2lsh
    int   rank,                         // MPI rank
    int   n,                            // number of data points
    int   d,                            // data dimension
    int   prime,                        // prime number
    int   m,                            // number of hash tables
    int   h,                            // number of concat hash functions
    float w,                            // bucket width
    const float *proj,                  // random projection, m*h*d
    const float *shift,                 // random shift, m*h
    const float *dataset,               // data set
    int   *hash_results);               // hash results (return)

} // end namespace clustering
