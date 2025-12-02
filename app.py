from agent_arxiv import run_workflow
from agent_arxiv.papers import collect_scored_papers, parse_score


def main():
    result = run_workflow("")
    scored_papers = collect_scored_papers(result)
    for paper in scored_papers:
        scores = parse_score(paper.get("score", "{}"))
        if scores != {}:
            impact = scores.get("impact", scores.get("technical_impact", 0))
            repro = scores.get("repro", scores.get("reproducibility", 0))
            potential = scores.get("potentiel", scores.get("potential", 0))
            originality = scores.get("originalite", scores.get("originality", 0))
            print(
                f"{paper['score_value']:.1f} - {paper['title']} "
                f"(Tech:{impact}/Repro:{repro}/Short:{potential}/Orig:{originality})"
            )
            print(f"     {paper['category']}")
            print(f"     {paper['url']}")
            print(f"     {paper['abstract']}")
            print(" ")

    linkedin_post = result.get("linkedin_post")
    if linkedin_post:
        print("Suggested LinkedIn post:\n")
        print(linkedin_post)


if __name__ == "__main__":
    main()
