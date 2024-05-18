# Authority-Rumor-Evidence Dataset (AuRED)

## Content of this repository
## Data
### Rumors
We provide AuRED and AuRED* data in JSON format files. Each file contains a list of JSON objects representing rumors. For each rumor, we provide the following entries:
```
{
  id [unique ID for the rumor]
  rumor [rumor tweet text]
  label [the veracity label of the rumor eithr SUPPORTS, REFUTES, NOT ENOUGH INFO]
  timeline [authorities timeline associated with the rumor each authority tweet is represented by authority Twitter account link, authority tweet ID, authority tweet text]
  evidence [authorities evidence tweets represented by authority Twitter account link, authority tweet ID, authority tweet text]
}
```
Examples:

```
{
  "id": "AuRED_089",
  "rumor": "وباء كورونا وصل الى الامارات 75 إصابة في ابوظبي و 63 إصابة في دبي  تحذير للامتناع عن السفر الى الامارات حفاظًا على السلامه و عدم نقل الوباء . اللهم أحفظ المسلمين في كل مكان..." ,
  "label": "REFUTES"
  "timeline": [["https://twitter.com/WHOEMRO", "1222971333522468867", "منظمة الصحة العالمية تعلن فاشية #فيروس_كورونا المستجد طارئة صحة عامة تثير قلقاً دوليا https://t.co/pVOXpZaPH7"],
   ["https://twitter.com/WHOEMRO", "1223608938136047616", "س. هل تحمي اللقاحات المضادة للالتهاب الرئوي من #فيروس_كورونا المستجد؟ ج. لا. لقاحات الالتهاب الرئوي لا تحمي من فيروس كورونا المستجد. هذا الفيروس جديد ومختلف ويحتاج لقاحاً خاصاً به. الباحثون يعملون على تطوير لقاح مضاد لهذا الفيروس. #اعرف_الحقائق https://t.co/QTGmI2flo9"],
   ["https://twitter.com/mohapuae", "1223361274618183681", "تعرف على أعراض فيروس كورونا الجديد #فيروس_كورونا_الجديد #فيروس_كورونا#كورونا#وزارة_الصحة_ووقاية_المجتمع_الإمارات https://t.co/jWALFtA68m"],
   ["https://twitter.com/mohapuae", "1223279618372882432", "مقتطفات من مشاركة وزارة الصحة ووقاية المجتمع في معرض ومؤتمر الصحة العربي2020 من خلال مجموعة من مبادرات ومشاريع الرعاية الصحية المبتكرة تحت شعار "صحة الإمارات مسؤولية مشتركة"#وزارة_الصحة_ووقاية_المجتمع_الإمارات#معرض_ومؤتمر_الصحة_العربي_2020#صحة_الإمارات https://t.co/c69pHj6ffd"],
   ......],
  "evidence": [["https://twitter.com/WHOEMRO","1222506828694794240","أكدت اليوم @WHO ظهور أولى حالات فيروس كورونا المستجد في إقليم شرق المتوسط، بالإمارات العربية المتحدة. عقب تأكيد @mohapuae في 29 يناير.
كان 4 أفراد من نفس العائلة من مدينة ووهان الصينية وصلوا إلى الإمارات في بداية يناير 2020، وتم إدخالهم المستشفى بعد تأكد إصابتهم ب #فيروس_كورونا."],
   ["https://twitter.com/mohapuae", "1222476311291142145", "إصابة أربعة أشخاص من عائلة صينية بفيروس كورونا الجديد جميعهم في حالة مستقرة وتم احتواؤهم وفق الإجراءات الاحترازية المعتمدة عالميا#وزارة_الصحة_ووقاية_المجتمع_الإمارات #فيروس_كورونا_الجديد #فيروس_كورونا https://t.co/ydy2esb20B"]
  ,....]
},
...,
{
  "id": "AuRED_105",
  "rumor": "تونس تعرض مساعدة ليبيا في علاج مصابي انفجار شاحنة الوقود في بنت بيّة #ليبيا #الشاهد للتفاصيل: https://t.co/s7fdU5fvgq" ,
  "label": "SUPPORTS",
  "timeline": [["https://twitter.com/NajlaElmangoush", "1554448728320344064", "أتقدم بالشكر لفخامة رئيس جمهورية #تونس السيد قيس سعيد @TnPresidencyعلى تضامنه وتسخير كل المستشفيات والأطقم الطبية لمساعدة جرحى #بنت_بيه وهذا التضامن يدل على أن ما يجمع الشعبين الشقيقين هو علاقات أخوية وروح تضامنية في السراء والضراء في كل الحالات @OJerandi 🇱🇾🇹🇳"],
  ["https://twitter.com/NajlaElmangoush", "1554027191788355584", "استفاقت بلدية #بنت_بية فجر اليوم على كارثة إنسانية وخبر مفزع، نتيجة انفجار صهريج الوقود، أسفر عن وفاة 5 أشخاص وإصابة قرابة 50 أخرين، أقدم تعازينا الحارة لأهالي المتوفيين، متمنيين الشفاء العاجل للمصابين، اللهم خفف عليهم مصابهم وثبت لهم الآجر."],
  ["https://twitter.com/Mofa_Libya", "1555688484396040193", "ندعوا المجتمع الدولي بالتحرك العاجل والفاعل لوقف التصعيد وتحمل مسؤوليته القانونية والأخلاقية إزاء الشعب الفلسطيني وتوفير الحماية له ، تجدد دولة #ليبيا موقفها الثابت من القضية الفلسطينية والحقوق المشروعة للشعب الفلسطيني الشقيق."],
  ["https://twitter.com/Mofa_Libya", "1555688334533558272", "تعرب وزارة الخارجية والتعاون الدولي بحكومة الوحدة الوطنية عن إدانتها واستنكارها الشديدين للعدوان الإسرائيلي على قطاع غزة مما أسفر عن سقوط شهداء وجرحي بينهم نساء وأطفال. https://t.co/Ijg2BG6F1p"],......],
  "evidence": [["https://twitter.com/Mofa_Libya", "1554448815524139013", "RT @NajlaElmangoush: أتقدم بالشكر لفخامة رئيس جمهورية #تونس السيد قيس سعيد @TnPresidencyعلى تضامنه وتسخير كل المستشفيات والأطقم الطبية لمساعدة جرحى #بنت_بيه وهذا التضامن يدل على أن ما يجمع الشعبين الشقيقين هو علاقات أخوية وروح تضامنية في السراء والضراء في كل الحالات @OJerandi\n 🇱🇾🇹🇳"],
  ["https://twitter.com/Mofa_Libya", "1554446617356427266", "1/2 وزارة الخارجية والتعاون الدولي تعرب عن شكرها وامتنانها العميق لما أعلنت عنه دولة #تونس الشقيقة في بيانها الأخير الذي سخرت فيه مستشفياتها وأطقمها الطبية التونسية؛ لمساعدة الليبيين الذين أصيبوا في بلدية #بنت_بيه إثر إنفجار صهريج الوقود. https://t.co/oWRtH9T7IC"]....]
},
...
{
  "id": "AuRED_078",
  "rumor": "منظمة الصحة العالمية تدعو لوقف منح الجرعة الثانية من لقاحات كورونا حتى سبتمبر المقبل ما يسمح بايصال الجرعة الاولى من اللقاح للفئات الاكثر " ,
  "label": "NOT ENOUGH INFO",
  "timeline": [["https://twitter.com/DrTedros", "1421857856522002437", "RT @BahraintvNews: الجهود الوطنية للتصدي لفيروس كورونا في مملكة البحرين تبهر المدير العام لمنظمة الصحة العالمية خلال زيارته للمملكة .@WHO @DrTedros  @BDF_Hospital#وزارة_الإعلام  #bahrain  #كورونا_في_البحرين #كورونا  #البحرين #المنامة #صوت_الوطن_وعين_الحدث"],
  ["https://twitter.com/WHOEMRO", "1424064461611147274", "قم بزيارة صفحتنا الجديدة "شركاء في الصحة" عن المملكة العربية #السعودية، الشريك الاستراتيجي القديم ل @WHO وأحد أكبر المانحين، ذات السجل الحافل في دعم المبادرات الصحية العالمية المنقذة للحياة وعمليات الطوارئ.\n\nمعًا من أجل تحقيق #الصحة_للجميع_وبالجميع\n\nhttps://t.co/0WAwx9mtF1"],
  ["https://twitter.com/WHOEMRO", "1423416531392806920", "❌الادعاء:  ينبغي على كل من تلقى لقاح كوفيد-19 الامتناع عن أخذ أي نوع من أنواع التخدير.✅الحقيقة: في الوقت الحالي، لا توجد أدلة علمية تؤيد أن التخدير يهدد الحياة أو غير آمن للاستخدام بعد تلقي لقاح كوفيد-19.لمزيد من حقائق اللقاح:➡️https://t.co/K7QtTVvBOK https://t.co/eFnCoVF9Jq"],
  ["https://twitter.com/WHOEMRO", "1423261810082426886", "ليس من الأسلم أن تُعطي رضيعك بدائل لبن الأم إذا كنتِ مصابة بمرض #كوفيد_19 إصابةً مؤكدة أو مُشتبهًا فيها 🤱https://t.co/wgp0yMCGnM\n#الأسبوع_العالمي_للرضاعة_الطبيعية https://t.co/B58EIK215r"]...],
  "evidence": []
},

...

```

### Authorities tweets media
We provide the videos and images extracted from the authorities tweets.

### qrels
We provide the [relevance judgments](https://github.com/AuRED2024/AuRED/tree/main/data/qrels) for both AuRED and AuRED* to be used for scoring the evidence retrieval models.

## Rumors folds
We provide the rumors 5 folds used in our experiments. Each fold file containing 32 rumors.

## Annotation Guidelines for Evidence Extraction

## Benchmarks code
### Evaluation
We provide the [evaluation scorers](https://github.com/AuRED2024/AuRED/tree/main/code/evaluation) for both the evidence retrieval and the rumor verification models.

To run the scorer you need to install [Pyterrier](https://pyterrier.readthedocs.io/en/latest/)


