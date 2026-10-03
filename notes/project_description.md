# Overview
A single knee scan can reveal a dozen different problems. In this competition, you are tasked to build machine learning models that detect a defined set of clinically important abnormalities on knee MRI examinations.

## Start 
2 months ago
## Close 
23 days to go

## Merger & Entry
### Description
The knee is the most commonly injured and imaged joint in the body. Osteoarthritis alone affects an estimated 654 million people worldwide, while acute knee injuries account for 15 to 40 percent of all sports-related trauma. MRIs show clinicians ligaments, cartilage, menisci, and bone in detail, without exposing patients to radiation.

Reading those scans isn’t always straightforward. ACL and MCL tears, meniscal damage, cartilage loss, fractures, and other abnormalities can be subtle, and radiologists don’t always interpret them the same way. Access to musculoskeletal radiologists is also limited, especially outside major medical centers, leading to delays and inconsistent diagnoses.

In this competition, you will develop multimodal machine learning models to detect twelve clinically important knee abnormalities. You'll work with the first RSNA AI Challenge dataset that pairs every imaging study with its original radiology report, enabling your models to learn from both visual scans and written diagnostic text.

High-performing models can act as robust decision support tools, delivering the accuracy, consistency, and speed needed to elevate expert-level knee MRI interpretation and improve care across disparate clinic settings.

## Evaluation
Submissions are evaluated by the average area under the ROC curve between the predicted confidence scores and the observed targets across the twelve targets:


The final score is, in other words, the macro-averaged AUC ROC.

## Submission File example
For each row in the test set, you must predict a confidence score for each of the twelve target labels. The file should contain a header and have the following format:

StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
<uid_2>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
...
Timeline
July 30, 2026 - Start Date.
October 15, 2026 - Entry Deadline. You must accept the competition rules before this date in order to compete.
October 15, 2026 - Team Merger Deadline. This is the last day participants may join or merge teams.
October 22, 2026 - Final Submission Deadline.
November 5, 2026 - Winners' Requirement Deadline. This is the deadline for winners to submit to the host/Kaggle their training code, video and method description.
All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.