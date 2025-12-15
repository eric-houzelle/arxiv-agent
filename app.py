from agent_arxiv import run_workflow
from agent_arxiv.logger import get_logger
from agent_arxiv.papers import collect_scored_papers, parse_score


def main():
    logger = get_logger(__name__)
    result = run_workflow("")
    scored_papers = collect_scored_papers(result)
    for paper in scored_papers:
        scores = parse_score(paper.get("score", "{}"))
        if scores != {}:
            impact = scores.get("impact", scores.get("technical_impact", 0))
            repro = scores.get("repro", scores.get("reproducibility", 0))
            potential = scores.get("potentiel", scores.get("potential", 0))
            originality = scores.get("originalite", scores.get("originality", 0))
            logger.info(
                "%s - %s (Tech:%s/Repro:%s/Short:%s/Orig:%s)\n"
                "     %s\n"
                "     %s\n"
                "     %s\n",
                f"{paper['score_value']:.1f}",
                paper["title"],
                impact,
                repro,
                potential,
                originality,
                paper["category"],
                paper["url"],
                paper["abstract"],
            )

    linkedin_post = result.get("linkedin_post")
    if linkedin_post:
        logger.info("Suggested LinkedIn post:\n%s", linkedin_post)


if __name__ == "__main__":
    main()
