这个代码理论上是可以直接按里面的README.TXT跑起来的，但是针对的是流行音乐的和弦（更准确的说是lmd数据集里的多轨流行音乐MIDI），可能对于古典音乐，musicalian，单轨音乐不太适用（但可以试试）。

算法是模板匹配+动态规划平滑。首先识别哪些轨道是和弦轨道（是软标签，main.py第47行的weights），用和弦轨道组成的chroma和bass chroma去和模板匹配。和弦的转移考虑了流行音乐的节拍性质，在downbeat的时候比较宽松，在upbeat和节拍内的时候比较严格。关于怎么调整：

1. 首先要保证MIDI里的节拍信息（也就是meter和tempo的标记）是准确的，也就是说用pretty_midi获得的midi.get_beats()和midi.get_downbeats()需要是准确的。如果不准确的话，需要手动修正MIDI里的这些标记。
2. 和弦类别需要做相应的修改。在chord_class.py文件里现在都是流行和爵士的和弦标签，QUALITIES存的是可能的和弦类型，INVERSIONS里存的是需要考虑转位的和弦类型，以及每个类型的转位种类。需要根据音乐风格做相应的调整。

Junyan


安装
>> pip3 install -r requirements.txt

用法
>> python3 main.py ./example.mid ./example.out
可以参考main.py和example.sh

其他
和弦字典见chord_class.py

Junyan
