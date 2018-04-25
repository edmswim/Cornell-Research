### How To Run ###
python script.py **[arguments]**

### Command Line Arguments ###
**--task** : can either be ***classify*** or ***regression***
<br />
**--model**: can be ***simpleNN***, ***lstm***, ***svm***
<br />
**--participantIndependent**: can be ***True*** or ***False***
<br />
**--totalDays**: can be from ***1*** to ***8*** days. totalDays means current day + previous day. If you type **--totalDays 1**, that means the model is only looking at that current day. However, if you type **--totalDays 5**, that means the model is looking at that current day + previous 4 days.


### Examples ###
python script.py --task classify --model lstm --participantIndependent True --totalDays 8
