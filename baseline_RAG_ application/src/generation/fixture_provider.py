from __future__ import annotations

ABSTENTION = "I could not find this information in the provided documents."


class FixtureProvider:
    """Deterministic generation driven by the question and retrieved evidence."""

    @staticmethod
    def _sections(prompt: str) -> tuple[str, str]:
        question_marker = "QUESTION:"
        context_marker = "CONTEXT:"
        if question_marker not in prompt or context_marker not in prompt:
            return prompt.lower(), prompt.lower()

        question_block = prompt.split(question_marker, 1)[1]
        question, context = question_block.split(context_marker, 1)
        return question.strip().lower(), context.strip().lower()

    def generate(self, prompt: str) -> str:
        question, context = self._sections(prompt)

        if "annual leave" in question:
            if "20 days" in context and "employee_handbook.txt" in context:
                return "Full-time employees receive 20 days of annual leave [employee_handbook.txt]."
            return ABSTENTION

        if "remote-work" in question or "remote work" in question:
            if "twice per week" in context and "work_policy.txt" in context:
                return "Employees may work remotely twice per week [work_policy.txt]."
            return ABSTENTION

        if "inc-2025-00417" in question:
            if "identity operations" in context and "incident_register.txt" in context:
                return "INC-2025-00417 is owned by the Identity Operations team [incident_register.txt]."
            return ABSTENTION

        if "project atlas" in question:
            has_project_link = "data platforms" in context and "project_register.txt" in context
            has_division_leader = "priya sharma" in context and "organization_directory.txt" in context
            if has_project_link and has_division_leader:
                return (
                    "Priya Sharma leads the division responsible for Project Atlas "
                    "[project_register.txt; organization_directory.txt]."
                )
            return ABSTENTION

        return ABSTENTION
