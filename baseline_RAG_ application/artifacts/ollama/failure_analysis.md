# Baseline RAG Failure Analysis

## vocabulary_mismatch: vocab-001

**Question:** How many remote-work days are permitted?

**Observed answer:** I could not find this information in the provided documents.

**Failure reason:** expected fact was missing from the answer; expected source was not cited; model abstained despite an expected answer

**Retrieved chunks:** employee_handbook-0-000, work_policy-0-000, project_register-0-000

**Likely cause:** Baseline fixed chunking or dense-only retrieval did not provide sufficient evidence, or generation did not follow the grounding contract.

**Proposed improvement:** Add query expansion, hybrid retrieval, reranking, metadata filters, multi-step retrieval, or stricter grounded-generation controls.

## multi_hop: multihop-001

**Question:** Who leads the division responsible for Project Atlas?

**Observed answer:** Priya Sharma leads the Data Platforms division, which is responsible for Project Atlas.[organization_directory.txt]

**Failure reason:** expected source was not cited

**Retrieved chunks:** project_register-0-000, organization_directory-0-000, incident_register-0-000

**Likely cause:** Baseline fixed chunking or dense-only retrieval did not provide sufficient evidence, or generation did not follow the grounding contract.

**Proposed improvement:** Add query expansion, hybrid retrieval, reranking, metadata filters, multi-step retrieval, or stricter grounded-generation controls.
