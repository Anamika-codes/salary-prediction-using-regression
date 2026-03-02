def create_features(df):

    df["ExperienceSquared"] = df["YearsExperience"] ** 2
    df["SkillExperienceInteraction"] = (
        df["YearsExperience"] * df["SkillsScore"]
    )

    return df