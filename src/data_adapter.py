import pandas as pd


def simplify_job_title(title):

    title = title.lower()

    if "scientist" in title:
        return "Data Scientist"
    elif "engineer" in title:
        return "ML Engineer"
    elif "analyst" in title:
        return "Data Analyst"
    elif "manager" in title:
        return "Manager"
    else:
        return "Other"


def simplify_location(loc):

    high_salary = ["US", "CA", "GB", "DE", "AU"]

    if loc in high_salary:
        return "HighIncomeCountry"
    else:
        return "OtherCountry"


def load_and_prepare_data(path="data/ds_salaries.csv"):

    df = pd.read_csv(path)

    df = df[[
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "company_size",
        "company_location",
        "remote_ratio",
        "salary_in_usd"
    ]]

    # Rename columns
    df = df.rename(columns={
        "work_year": "WorkYear",
        "experience_level": "ExperienceLevel",
        "employment_type": "EmploymentType",
        "job_title": "JobRole",
        "company_size": "CompanySize",
        "company_location": "Location",
        "remote_ratio": "RemoteRatio",
        "salary_in_usd": "Salary"
    })

    # Reduce category explosion (VERY IMPORTANT)
    df["JobRole"] = df["JobRole"].apply(simplify_job_title)
    df["Location"] = df["Location"].apply(simplify_location)

    # Experience mapping
    exp_map = {"EN": 1, "MI": 4, "SE": 7, "EX": 12}
    df["YearsExperience"] = df["ExperienceLevel"].map(exp_map)

    # Better skill proxy
    df["SkillsScore"] = df["YearsExperience"] * 10

    df.drop(columns=["ExperienceLevel"], inplace=True)

    return df