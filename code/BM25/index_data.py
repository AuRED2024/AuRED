declare -a claims=("AuRED_005" "AuRED_012" "AuRED_014" "AuRED_019" "AuRED_037" "AuRED_038" "AuRED_041" "AuRED_051" 
                 "AuRED_072" "AuRED_073" "AuRED_079" "AuRED_083" "AuRED_084" "AuRED_085" "AuRED_087" "AuRED_095"
                 "AuRED_096" "AuRED_097" "AuRED_098" "AuRED_099" "AuRED_100" "AuRED_103" "AuRED_130" "AuRED_131" "AuRED_133"
                 "AuRED_134" "AuRED_135" "AuRED_136" "AuRED_137" "AuRED_138" "AuRED_139" "AuRED_142" "AuRED_143" "AuRED_144" "AuRED_145"
                 "AuRED_147" "AuRED_148" "AuRED_149" "AuRED_150" "AuRED_151" "AuRED_152" "AuRED_153" "AuRED_154" "AuRED_156" "AuRED_157"
                 "AuRED_159" "AuRED_160" "AuRED_105" "AuRED_106" "AuRED_107" "AuRED_108" "AuRED_109" "AuRED_110" "AuRED_111" "AuRED_112" 
                 "AuRED_114" "AuRED_115" "AuRED_116" "AuRED_119" "AuRED_121" "AuRED_123" "AuRED_124" "AuRED_125" "AuRED_127" "AuRED_128"
                 "AuRED_129" "AuRED_001" "AuRED_004" "AuRED_006" "AuRED_010" "AuRED_013" "AuRED_015" "AuRED_016" "AuRED_017" "AuRED_020" "AuRED_021" 
                 "AuRED_023" "AuRED_024" "AuRED_027" "AuRED_028" "AuRED_029" "AuRED_030" "AuRED_032" "AuRED_033" "AuRED_034" "AuRED_035" "AuRED_036"
                 "AuRED_039" "AuRED_045" "AuRED_046" "AuRED_047" "AuRED_049" "AuRED_050" "AuRED_052" "AuRED_054" "AuRED_055" "AuRED_056" "AuRED_058" 
                 "AuRED_060" "AuRED_061" "AuRED_063" "AuRED_066" "AuRED_069" "AuRED_070" "AuRED_071" "AuRED_074" "AuRED_075" "AuRED_076" "AuRED_081" 
                 "AuRED_082" "AuRED_088" "AuRED_092" "AuRED_022" "AuRED_025" "AuRED_040" "AuRED_042" "AuRED_044" "AuRED_062" "AuRED_065" "AuRED_067"
                 "AuRED_089" "AuRED_091" "AuRED_093" "AuRED_094" "AuRED_102" "AuRED_117" "AuRED_120" "AuRED_155" "AuRED_002" "AuRED_007" "AuRED_011" 
                 "AuRED_043" "AuRED_064" "AuRED_068" "AuRED_086" "AuRED_090" "AuRED_101" "AuRED_104" "AuRED_113" "AuRED_132" "AuRED_140" "AuRED_141" 
                 "AuRED_146" "AuRED_158" "AuRED_003" "AuRED_008" "AuRED_009" "AuRED_018" "AuRED_026" "AuRED_031" "AuRED_048" "AuRED_053" "AuRED_057" 
                 "AuRED_059" "AuRED_077" "AuRED_078" "AuRED_080" "AuRED_118" "AuRED_122" "AuRED_126"))

for claim in ${claims[@]} ; do
  python -m pyserini.index.lucene \
   --collection JsonCollection \
   --input ./AuRED_JSON/${claim} \
   --language ar \
   --index AuRED_indexes/${claim}_index \
   --generator DefaultLuceneDocumentGenerator \
   --threads 1 \
   --storePositions --storeDocvectors --storeRaw
done
