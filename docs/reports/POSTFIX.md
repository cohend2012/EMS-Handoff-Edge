# Post-fix run — HANDOFF.AI pipeline over the MoCo call-set corpus

| | |
|---|---|
| start / end | 2026-10-24 17:04 -> 18:04 PDT (main batch 55.6 min, then two gap passes for targets that failed while Ollama restarted) |
| app repo | ~/EMS-Handoff-app, branch fix/post-baseline |
| commit | 163d549 for the main batch; 7961c5a (tooling commits only, no pipeline change) for the gap passes |
| model | qwen3:4b-instruct, pinned through OLLAMA_MODEL; every accepted report has _meta.model == pinned model |
| transcription (secondary path) | Groq whisper-large-v3-turbo via the app's auto transcriber |
| inputs | 22 call-sets (~/callsets/CS*/combined_transcript.txt), all 51 bursts (~/segments/*.wav), 11 fragments under 15 words |
| order | the baseline's: set IDs sorted, then bursts in callsets.csv order |
| runner | scripts/corpus/stage5_reports.py --all --out ~/reports_postfix --log ~/labels/stage5_postfix_log.csv |
| outputs | 73 reports in this folder (22 combined + 51 bursts); nothing written into ~/callsets or ~/reports |

## Incidents during the run

Ollama itself died and restarted three times (systemd NRestarts 3; about 0.2 GB free with 1.6 GB swap in use
while the model server holds ~5.6 GB). Four runs failed on those restarts: CS012_combined (17:15, Ollama
unreachable), S040_01 (17:30 and again 18:01, 'model runner has unexpectedly stopped'), S055_01 (17:45,
unreachable). The pipeline code did not fail. Re-running the same command fills only the gaps (existing outputs
are skipped): gap pass 1 at 18:00 did CS012_combined and S055_01, gap pass 2 at 18:03 did S040_01.
No wrong-model rejections (the pin held), no refusals, no timeouts.

## Comparison with the baseline (~/reports)

scripts/corpus/compare_runs.py ~/reports ~/reports_postfix

```
                             reports             reports_postfix   
folder                       /home/voxai/reports  /home/voxai/reports_postfix
run log                      /home/voxai/labels/stage5_log.csv  /home/voxai/labels/stage5_postfix_log.csv
reports present              45/73               73/73             

CRASHES AND REFUSALS (from the run logs; a target that later succeeded is still listed)
crashes (failed/timeout)     2                   4                 
wrong-model rejections       7                   0                 
refusals (too short)         2                   0                 
UNCLEAR reports              0                   20                
missing reports              28                  0                 
  reports: failed   CS022_combined   AttributeError: 'NoneType' object has no attribute 'extend'
  reports: failed   S007_01          AttributeError: 'NoneType' object has no attribute 'extend'
  reports: failed   CS015_combined   Report failed: The report is too short to process. Please say or paste more detail.
  reports: failed   CS018_combined   Report failed: The report is too short to process. Please say or paste more detail.
  reports: missing CS018_combined, CS022_combined, S045_01, S046_01, S043_01, S037_01, S039_01, S041_01, S034_01, S035_01, S036_01, S027_01, S029_01, S030_01, S031_01, S032_01, S020_01, S016_01, S055_01, S013_01, S014_01, S012_01, S009_01, S007_01, S051_01, S052_01, S005_01, S003_01
  reports_postfix: failed   CS012_combined   Report failed: ollama failed after 3 tries: cannot reach Ollama at http://localhost:11434/v1. Is 'ollama serve (succeeded later)
  reports_postfix: failed   S040_01          Report failed: ollama error HTTP 500: Error code: 500 - {'error': {'message': 'an error was encountered while  (succeeded later)
  reports_postfix: failed   S055_01          Report failed: ollama failed after 3 tries: cannot reach Ollama at http://localhost:11434/v1. Is 'ollama serve (succeeded later)
  reports_postfix: failed   S040_01          Report failed: ollama error HTTP 500: Error code: 500 - {'error': {'message': 'model runner has unexpectedly s (succeeded later)
  reports_postfix: UNCLEAR S045_01, S046_01, S043_01, S037_01, S039_01, S038_01, S034_01, S036_01, S030_01, S031_01, S032_01, S055_01, S013_01, S014_01, S012_01, S051_01, S052_01, S002_01, S005_01, S003_01

HALLUCINATION CANDIDATES (forward claims checked against the transcript)
  claims found: yes          230                 250               
  claims found: partial      43                  36                
  claims found: no           120                 98                
  claims total               393                 384               
  'no' in ems_interventions  14                  13                
  'no' in history.medications 25                  0                 
  'no' in history.past_medical 1                   0                 
  'no' in history.allergies  36                  50                
  'no' in history.code_status 34                  23                
  'no' in vital_signs        1                   1                 
  'no' in patient.age        2                   2                 
  'no' in eta_minutes        7                   9                 
  over the 45 reports present in both folders:
    claims found: yes        230                 224               
    claims found: partial    43                  34                
    claims found: no         120                 67                

FIELDS FILLED (25 clinical fields; over the 45 reports present in both folders)
  mean per report            15.9                14.9              
  total                      714                 669               

CONFIDENCE (data_quality.confidence)
  combined HIGH              1                   2                 
  combined MEDIUM            19                  20                
  burst HIGH                 2                   2                 
  burst MEDIUM               21                  29                
  burst LOW                  2                   20                

PER-SET DIFFS (alert_type / sex / age; reports -> reports_postfix; * = changed, - = empty, missing = no report)
  set    report          alert_type               sex              age                     
* CS001  CS001_combined  STROKE                   - -> M           -                       
  CS002  CS002_combined  STROKE                   F                68                      
* CS003  CS003_combined  STROKE                   - -> M           -                       
  CS004  CS004_combined  TRAUMA                   M                46                      
* CS005  CS005_combined  TRAUMA                   - -> M           -                       
  CS006  CS006_combined  STROKE                   M                83                      
  CS007  CS007_combined  CARDIAC                  M                59                      
  CS008  CS008_combined  CARDIAC                  F                62                      
  CS009  CS009_combined  TRAUMA                   M                23                      
  CS010  CS010_combined  STROKE                   M                60                      
  CS011  CS011_combined  TRAUMA                   F                100                     
  CS012  CS012_combined  ENDOCRINE                M                35                      
  CS013  CS013_combined  ENDOCRINE                M                22 months               
  CS014  CS014_combined  PEDIATRIC                F                3 months                
  CS016  CS016_combined  TRAUMA                   M                66                      
  CS017  CS017_combined  CARDIAC                  F                76                      
* CS018  CS018_combined  missing -> STROKE        missing -> -     missing -> -            
  CS019  CS019_combined  STROKE                   M                71                      
  CS020  CS020_combined  CARDIAC                  F                40                      
  CS021  CS021_combined  ENDOCRINE                M                72                      
* CS022  CS022_combined  missing -> STROKE        missing -> M     missing -> 61           
  CS023  CS023_combined  STROKE                   F                75                      
* CS001  S044_01         STROKE                   - -> M           -                       
* CS001  S045_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS001  S046_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS002  S043_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS003  S037_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS003  S039_01         missing -> UNCLEAR       missing -> M     missing -> -            
* CS003  S041_01         missing -> STROKE        missing -> -     missing -> -            
* CS004  S038_01         MEDICAL -> UNCLEAR       -                -                       
* CS005  S033_01         STROKE -> TRAUMA         - -> M           -                       
* CS005  S034_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS005  S035_01         missing -> MEDICAL       missing -> -     missing -> -            
* CS005  S036_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS006  S027_01         missing -> STROKE        missing -> M     missing -> 83           
* CS006  S029_01         missing -> STROKE        missing -> M     missing -> -            
* CS006  S030_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS006  S031_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS006  S032_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS010  S020_01         missing -> STROKE        missing -> -     missing -> 60           
* CS010  S021_01         ENDOCRINE                - -> M           -                       
* CS013  S016_01         missing -> PEDIATRIC     missing -> M     missing -> 22 months    
* CS013  S017_01         ENDOCRINE                - -> M           -                       
* CS014  S055_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS014  S013_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS014  S014_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS016  S012_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS018  S009_01         missing -> STROKE        missing -> -     missing -> -            
* CS020  S007_01         missing -> CARDIAC       missing -> F     missing -> 40           
* CS021  S049_01         ENDOCRINE                - -> M           -                       
* CS021  S050_01         ENDOCRINE                - -> M           -                       
* CS021  S051_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS021  S052_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS021  S002_01         MEDICAL -> UNCLEAR       -                -                       
* CS022  S048_01         MEDICAL -> STROKE        M                61                      
* CS023  S005_01         missing -> UNCLEAR       missing -> -     missing -> -            
* CS023  S003_01         missing -> UNCLEAR       missing -> F     missing -> -            
```

## Four sample cases, scored against tests/gold (scripts/benchmark.py)

baseline = ~/reports/samples_baseline (handoff-app abff900), fix = ~/reports_postfix/samples (fix/post-baseline).

```
case          weighted        recall     precision           hit      omission         wrong     abstained hallucination     extract_s
           base -> fix   base -> fix   base -> fix   base -> fix   base -> fix   base -> fix   base -> fix   base -> fix   base -> fix
quick     69.0 -> 83.3  100.0 -> 100.0  62.5 -> 73.3     10 -> 11        0 -> 0         0 -> 0         3 -> 4         6 -> 4    48.951 -> 45.461
stemi     96.0 -> 98.0   90.0 -> 95.0   90.0 -> 95.0     18 -> 19        0 -> 0         2 -> 1         2 -> 2         0 -> 0    69.122 -> 52.093
trauma    90.0 -> 94.0  100.0 -> 100.0  88.9 -> 94.1     16 -> 16        0 -> 0         0 -> 0         3 -> 4         2 -> 1     57.84 -> 59.283
sepsis    79.1 -> 79.1   84.2 -> 84.2   88.9 -> 88.9     16 -> 16        1 -> 1         2 -> 2         0 -> 0         0 -> 0    53.791 -> 55.318
mean      83.5 -> 88.6 
```
