"""
Predefined job seeker profiles for the User Simulator.

Each profile specifies demographic info, experience, preferences,
and behavioral tendencies used to generate realistic multi-turn dialogues.
"""

SAMPLE_PROFILES = [
    {
        "id": "profile_001",
        "gender": "female",
        "age": 28,
        "years_of_experience": 5,
        "current_role": "Software Engineer",
        "desired_role": "Senior Software Engineer",
        "skills": ["Python", "Java", "Machine Learning"],
        "salary_expectation": "25k-35k",
        "location_preference": "Beijing",
        "personality": "cooperative",
        "notes": "Open to negotiation, values work-life balance",
    },
    {
        "id": "profile_002",
        "gender": "male",
        "age": 35,
        "years_of_experience": 10,
        "current_role": "Tech Lead",
        "desired_role": "Engineering Manager",
        "skills": ["System Design", "Team Management", "Go", "Kubernetes"],
        "salary_expectation": "40k-55k",
        "location_preference": "Shanghai",
        "personality": "assertive",
        "notes": "Has competing offers, may reject if conditions not met",
    },
    {
        "id": "profile_003",
        "gender": "female",
        "age": 23,
        "years_of_experience": 1,
        "current_role": "Junior Data Analyst",
        "desired_role": "Data Scientist",
        "skills": ["Python", "SQL", "Tableau"],
        "salary_expectation": "12k-18k",
        "location_preference": "Shenzhen",
        "personality": "hesitant",
        "notes": "Uncertain about career direction, needs guidance",
    },
    {
        "id": "profile_004",
        "gender": "male",
        "age": 40,
        "years_of_experience": 15,
        "current_role": "Principal Architect",
        "desired_role": "CTO",
        "skills": ["Architecture", "Cloud", "Strategy", "AI/ML"],
        "salary_expectation": "60k-80k",
        "location_preference": "Remote",
        "personality": "demanding",
        "notes": "Very selective, only interested in top-tier companies",
    },
    {
        "id": "profile_005",
        "gender": "female",
        "age": 30,
        "years_of_experience": 7,
        "current_role": "Product Manager",
        "desired_role": "Senior Product Manager",
        "skills": ["Product Strategy", "User Research", "Agile"],
        "salary_expectation": "30k-40k",
        "location_preference": "Hangzhou",
        "personality": "cooperative",
        "notes": "Interested in AI product roles, flexible on salary",
    },
]


def get_profile_prompt(profile: dict) -> str:
    """Convert a profile dict into a natural language prompt for the simulator."""
    skills_str = ", ".join(profile["skills"])
    return (
        f"You are a job seeker with the following background:\n"
        f"- Gender: {profile['gender']}\n"
        f"- Age: {profile['age']}\n"
        f"- Years of experience: {profile['years_of_experience']}\n"
        f"- Current role: {profile['current_role']}\n"
        f"- Desired role: {profile['desired_role']}\n"
        f"- Key skills: {skills_str}\n"
        f"- Salary expectation: {profile['salary_expectation']} RMB/month\n"
        f"- Location preference: {profile['location_preference']}\n"
        f"- Personality: {profile['personality']}\n"
        f"- Additional notes: {profile['notes']}\n\n"
        f"Respond naturally as this person in a recruitment conversation. "
        f"Stay in character and express preferences consistent with your profile."
    )
