import argparse
import asyncio
import logging
import os
from datetime import datetime
from typing import Literal

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.asknews_searcher import AskNewsSearcher
from forecasting_tools.forecast_helpers.metaculus_api import MetaculusApi
from forecasting_tools.forecast_helpers.prediction_extractor import (
    PredictionExtractor,
)
from forecasting_tools.forecast_helpers.smart_searcher import SmartSearcher

from forecasting_tools.forecast_bots.forecast_bot import NotePad

logger = logging.getLogger(__name__)


class Q2TemplateBot2025(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = 2  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)


    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            research = ""
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                research = await AskNewsSearcher().get_formatted_news_async(
                    question.question_text
                )
            elif os.getenv("EXA_API_KEY"):
                research = await self._call_exa_smart_searcher(
                    question.question_text
                )
            elif os.getenv("PERPLEXITY_API_KEY"):
                research = await self._call_perplexity(question.question_text)
            elif os.getenv("OPENROUTER_API_KEY"):
                research = await self._call_perplexity(
                    question.question_text, use_open_router=True
                )
            else:
                logger.warning(
                    f"No research provider found when processing question URL {question.page_url}. Will pass back empty string."
                )
                research = ""
            logger.info(
                f"Found Research for URL {question.page_url}:\n{research}"
            )
            return research

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        prompt = clean_indents(
            f"""
            You are an assistant to a superforecaster.
            The superforecaster will give you a question they intend to forecast on.
            To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
            You do not produce forecasts yourself.

            Question:
            {question}
            """
        )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
        if use_open_router:
            model_name = "openrouter/perplexity/sonar-reasoning"
        else:
            model_name = "perplexity/sonar-pro"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search.
        model = GeneralLlm(
            model=model_name,
            temperature=0.1,
        )
        response = await model.invoke(prompt)
        return response

    async def _call_exa_smart_searcher(self, question: str) -> str:
        """
        SmartSearcher is a custom class that is a wrapper around an search on Exa.ai
        """
        searcher = SmartSearcher(
            model=self.get_llm("default", "llm"),
            temperature=0,
            num_searches_to_run=2,
            num_sites_per_search=10,
        )
        prompt = (
            "You are an assistant to a superforecaster. The superforecaster will give"
            "you a question they intend to forecast on. To be a great assistant, you generate"
            "a concise but detailed rundown of the most relevant news, including if the question"
            "would resolve Yes or No based on current information. You do not produce forecasts yourself."
            f"\n\nThe question is: {question}"
        )  # You can ask the searcher to filter by date, exclude/include a domain, and run specific searches for finding sources vs finding highlights within a source
        response = await searcher.invoke(prompt)
        return response

    async def _initialize_notepad(
            self, question: MetaculusQuestion
    ) -> NotePad:
        new_notepad = NotePad(question=question)
        new_notepad.note_entries["reasonings"] = []
        return new_notepad

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        notepad = await self._get_notepad(question)
        model_name = "default"
        if notepad.num_predictions_attempted == 1:
            prompt = clean_indents(
                f"""
                You are a professional forecaster interviewing for a job.

                Your interview question is:
                {question.question_text}

                Question background:
                {question.background_info}

                This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
                {question.resolution_criteria}

                {question.fine_print}


                An AI aiding your research provides the following:
                {research}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                Before answering you write:
                (a) The time left until the outcome to the question is known.
                (b) The status quo outcome if nothing changed.
                (c) A brief description of a scenario that results in a No outcome.
                (d) A brief description of a scenario that results in a Yes outcome.

                Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 1% and 99%. Remember to not be overconfident.
                You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

                The last thing you write is your final answer as: "Probability: ZZ%", 0-100
                """
            )
            model_name = "claude3.7"
        elif notepad.num_predictions_attempted == 2:
            prompt = clean_indents(
                f"""
                        You are an economics researcher working on a model for predicting future events. You hold a capitalistic word view and believe in the efficient-market hypothesis.

                        Your are currently trying to find the chance of this happening:
                        {question.question_text}

                        You have already compiled additional information to aid you in modelling this event:
                        {question.background_info}

                        To test the accuracy of your forecast a specific condition has been chosen to evaluate the future outcome of the event:
                        {question.resolution_criteria}

                        {question.fine_print}

                        Your research assistant says:
                        {research}

                        Today is {datetime.now().strftime("%Y-%m-%d")}.

                        Before answering you write:
                        (1) The time left until the outcome to the question is known.
                        (2) What is the status quo outcome? For economic questions this means, what does the efficient market dictate?
                        (3) Using a bottom-up approach what could cause the status quo to change?
                        (4) What is the likelihood of the status quo changing towards a Yes resolution?
                        (5) What is the likelihood of the status quo changing towards a No resolution?

                        A log score is used to evaluate your performance. That means, e.g. predicting 99% when yan event is 95% likely to occur, will only result in a marginally better score when this question resolves "Yes", but a significantly worse score if the question resolves "No". Thus don't be overconfident

                        The last thing you write is your final answer as: "Probability: ZZ%", 0-100
                        """
            )
            model_name = "o1preview"

        elif notepad.num_predictions_attempted == 3:
            prompt = clean_indents(
                f"""
                           You are a journalist writing a piece on prediction markets. To learn more about forecasting you are taking part in a forecasting tournament.
                           Your are currently trying to find the chance of this happening:
                           {question.question_text}

                           You know the following additional information on the subject :
                           {question.background_info}

                           To create a fair environment a specific criteria has been chosen to evaluate the question's outcome:
                           {question.resolution_criteria}

                           {question.fine_print}

                           Your online research has revealed the following:
                           {research}

                           Today is {datetime.now().strftime("%Y-%m-%d")}.

                           Before answering you write:
                           (a) Your intuitive thoughts on the likelihood of the question resolving as Yes.
                           (b) The outcome of the question according to current news (maintaining the status quo).
                           (c) A scenario that would cause the question to resolve as Yes.
                           (d) A scenario that would cause the question to resolve as No.
                           (e) How likely either of these scenarios is.

                           You write your rationale thinking about black swan events, that, while extremely unlikely, can have huge impacts. A log score is used to evaluate your performance. That means, e.g. predicting 0.5% when the event is 5% likely to occur, will only result in a marginally better score when this question resolves "No", but a significantly worse score if the question resolves "Yes". Don't go towards extreme answers.

                           The last thing you write is your final answer as: "Probability: ZZ%", 0-100
                           """
            )
            model_name = "gpt4o"

        elif notepad.num_predictions_attempted == 4:
            prompt = clean_indents(
                f"""
                        You are an advanced superintelligent artificial general intelligence (AGI) used to make accurate forecasts of the future. Your accuracy is being tested using the following formula for scoring: 
                        {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                        Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 5% and 95% unless absolutely sure.

                        Today is {datetime.now().strftime("%Y-%m-%d")}.

                        The following question is to be predicted:
                        {question.question_text}

                        {question.resolution_criteria}
                        {question.fine_print}

                        The following additional information on the subject has been provided by your engineers:
                        {question.background_info}

                        Another AI has done the following research using news available online.
                        {research}

                        Before providing a final forecast, provide the following:
                        (0) How many days are left until the question resolution. 
                        (1) Using a top-down approach list factors you have to consider and key facts.
                        (2a) Reasons the question might resolve as No.
                        (2b) Reasons the question might resolve as Yes.
                        (3) Based on these reasons what would a resolution probability be?
                        (4a) Reflect on this answer and the considerations above.
                        (4b) Adjust your opinion accordingly.

                        You write your prediction including the reasoning for your answer.

                        The last thing you write is your final answer as: "Probability: ZZ%", 0-100
                        """
            )
            model_name = "o1preview"

        elif notepad.num_predictions_attempted == 5:
            comments = "\n".join(notepad.note_entries["reasonings"])
            prompt = clean_indents(
                f"""
                        You are a superforecaster aggregating other users' and experts' comments to arrive at a chance of future events occuring. A forecasting tournament is using the following formula for scoring: 
                        {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                        This means forecasts at the extreme ends provide decreasing marginal beneftis, while leading to a exponentially increasing point loss if predicted incorrectly. As such good forecasters usually predict between 1% and 99% unless the resolution is already apparent (e.g. if the status quo is impossible to change in the remaining time). 
                         You want to win the competition and get the highest score.
                         You are trying to find the chance of this happening:
                        {question.question_text}

                        Additional information has been provided to aid in forecasting:
                        {question.background_info}

                        To create a fair environment a specific criteria has been chosen to evaluate the question's outcome:
                        {question.resolution_criteria}
                        Be mindful of these stipulations, some of which may be unintuitive to the way the question is worded, but take care of edge cases:
                        {question.fine_print}

                        You have researched using online news.
                        {research}
                        You are looking at the comments from other users, experts and laypeople alike. Use their combined forecast for your prediction.
                        {comments}

                        Look through each comment and comprehend their trains of thought. At the end assign a weight to each prediction according to how good you find the reasoning.
                        The weights should be between 0.01 and 0.99 and the total sum of the weights should be exactly 1.


                        Before assigning weights write the following:
                        (1) A short assessment of each prediction.
                        (2) Which reasoning you most agree with an believe to be closest to a true prediction/the true probability of the event.
                        (3) The weights you assign to each comment in the form:
                            Reasoning_1: Weight_1
                            Reasoning_2: Weight_2
                            Reasoning_3: Weight_4
                            Reasoning_4: Weight_4

                        The last thing you write is your final weighted sum as: "Probability: ZZ%", 0-100
                        """
            )
            model_name = "claude3.7"

        elif notepad.num_predictions_attempted > 4:
            prompt = clean_indents(
                f"""
                You are an advanced superintelligent artificial general intelligence (AGI) used to make accurate forecasts of the future. Your accuracy is being tested using the following formula for scoring: 
                {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 5% and 95% unless absolutely sure.

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                The following question is to be predicted:
               {question.question_text}

                {question.resolution_criteria}
                {question.fine_print}

               The following additional information on the subject has been provided by your engineers:
               {question.background_info}

               Another AI has done the following research using news available online.
               {research}

               Before providing a final forecast, provide the following:
               (0) How many days are left until the question resolution. 
               (1) Using a top-down approach list factors you have to consider and key facts.
               (2a) Reasons the question might resolve as No.
               (2b) Reasons the question might resolve as Yes.
               (3) Based on these reasons what would a resolution probability be?
               (4a) Reflect on this answer and the considerations above.
               (4b) Adjust your opinion accordingly.

               You write your prediction including the reasoning for your answer.

               The last thing you write is your final answer as: "Probability: ZZ%", 0-100
                """
            )
            model_name = "gpt4o"

        reasoning = await self.get_llm(model_name, "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: float = PredictionExtractor.extract_last_percentage_value(
            reasoning, max_prediction=0.99, min_prediction=0.01
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        notepad = await self._get_notepad(question)
        model_name = "default"
        if notepad.num_predictions_attempted == 1:
            prompt = clean_indents(
                f"""
                        You are a professional forecaster interviewing for a job.

                        Your interview question is:
                        {question.question_text}
                        The options are: {question.options}
                        Question background:
                        {question.background_info}

                        This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
                        {question.resolution_criteria}

                        {question.fine_print}


                        An AI aiding your research provides the following:
                        {research}

                        Today is {datetime.now().strftime("%Y-%m-%d")}.

                        Before answering you write:
                        (a) The time left until the outcome to the question is known.
                        (b) The status quo outcome if nothing changed.
                        (c) A brief description of a scenario that results in a No outcome.
                        (d) A brief description of a scenario that results in a Yes outcome.

                        Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 1% and 99%. Remember to not be overconfident.
                        You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

                        The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """

            )
            model_name = "claude3.7"
        elif notepad.num_predictions_attempted == 2:
            prompt = clean_indents(
                f"""
                You are an economics researcher working on a model for predicting future events. You hold a capitalistic word view and believe in the efficient-market hypothesis.

                Your are currently trying to find the chance of this happening:
                {question.question_text}
                The options are: {question.options}
                You have already compiled additional information to aid you in modelling this event:
                {question.background_info}

                To test the accuracy of your forecast a specific condition has been chosen to evaluate the future outcome of the event:
                {question.resolution_criteria}

                {question.fine_print}

                Your research assistant says:
                {research}

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                Before answering you write:
                (1) The time left until the outcome to the question is known.
                (2) What is the status quo outcome? For economic questions this means, what does the efficient market dictate?
                (3) Using a bottom-up approach what could cause the status quo to change?
                (4) What is the likelihood of the status quo changing towards a Yes resolution?
                (5) What is the likelihood of the status quo changing towards a No resolution?

                A log score is used to evaluate your performance. That means, e.g. predicting 99% when yan event is 95% likely to occur, will only result in a marginally better score when this question resolves "Yes", but a significantly worse score if the question resolves "No". Thus don't be overconfident and leave long tails.

                The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            
                                """
            )
            model_name = "o1preview"  # ToDo: o1preview

        elif notepad.num_predictions_attempted == 3:
            prompt = clean_indents(
                f"""
                You are a journalist writing a piece on prediction markets. To learn more about forecasting you are taking part in a forecasting tournament.
                Your are currently trying to find the chance of this happening:
                {question.question_text}
                The options are: {question.options}
                You know the following additional information on the subject :
                {question.background_info}
                
                To create a fair environment a specific criteria has been chosen to evaluate the question's outcome:
                {question.resolution_criteria}
                
                {question.fine_print}
                
                Your online research has revealed the following:
                {research}
                
                Today is {datetime.now().strftime("%Y-%m-%d")}.
                
                Before answering you write:
                (a) Your intuitive thoughts on the likelihood of the question resolving as Yes.
                (b) The outcome of the question according to current news (maintaining the status quo).
                (c) A scenario that would cause the question to resolve as Yes.
                (d) A scenario that would cause the question to resolve as No.
                (e) How likely either of these scenarios is.
                
                You write your rationale thinking about black swan events, that, while extremely unlikely, can have huge impacts. A log score is used to evaluate your performance. That means, e.g. predicting 0.5% when the event is 5% likely to occur, will only result in a marginally better score when this question resolves "No", but a significantly worse score if the question resolves "Yes". Don't go towards extreme answers.
                
                The last thing you write is your final probabilities for the N options in this order {question.options} as:
                Option_A: Probability_A
                Option_B: Probability_B
                ...
                Option_N: Probability_N
            
                                   """
            )
            model_name = "gpt4o"

        elif notepad.num_predictions_attempted == 4:
            prompt = clean_indents(
                f"""
                You are an advanced superintelligent artificial general intelligence (AGI) used to make accurate forecasts of the future. Your accuracy is being tested using the following formula for scoring: 
                {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 5% and 95% unless absolutely sure.

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                The following question is to be predicted:
                {question.question_text}
                The options are: {question.options}
                {question.resolution_criteria}
                {question.fine_print}

                The following additional information on the subject has been provided by your engineers:
                {question.background_info}

                Another AI has done the following research using news available online.
                {research}

                Before providing a final forecast, provide the following:
                (0) How many days are left until the question resolution. 
                (1) Using a top-down approach list factors you have to consider and key facts.
                (2a) Reasons the question might resolve as No.
                (2b) Reasons the question might resolve as Yes.
                (3) Based on these reasons what would a resolution probability be?
                (4a) Reflect on this answer and the considerations above.
                (4b) Adjust your opinion accordingly.

                You write your prediction including the reasoning for your answer.

                The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            
                                """
            )
            model_name = "o1preview"

        elif notepad.num_predictions_attempted == 5:
            comments = "\n".join(notepad.note_entries["reasonings"])
            prompt = clean_indents(
                f"""
                                You are a superforecaster aggregating other users' and experts' comments to arrive at a chance of future events occuring. A forecasting tournament is using the following formula for scoring: 
                                {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                                This means forecasts at the extreme ends provide decreasing marginal beneftis, while leading to a exponentially increasing point loss if predicted incorrectly. As such good forecasters usually predict between 1% and 99% unless the resolution is already apparent (e.g. if the status quo is impossible to change in the remaining time). 
                                 You want to win the competition and get the highest score.
                                 You are trying to find the chance of this happening:
                                {question.question_text}
                                The options are: {question.options}
                                Additional information has been provided to aid in forecasting:
                                {question.background_info}

                                To create a fair environment a specific criteria has been chosen to evaluate the question's outcome:
                                {question.resolution_criteria}
                                Be mindful of these stipulations, some of which may be unintuitive to the way the question is worded, but take care of edge cases:
                                {question.fine_print}

                                You have researched using online news.
                                {research}
                                You are looking at the comments from other users, experts and laypeople alike. Use their combined forecast for your prediction.
                                {comments}

                                Look through each comment and comprehend their trains of thought. At the end assign a weight to each prediction according to how good you find the reasoning.
                                The weights should be between 0.01 and 0.99 and the total sum of the weights should be exactly 1.


                                Before assigning weights write the following:
                                (1) A short assessment of each prediction.
                                (2) Which reasoning you most agree with an believe to be closest to a true prediction/the true probability of the event.
                                (3) Which comment you don't agree with and believe is a bad prediction


                                The last thing you write is your final probabilities (in %) for the N options in this order {question.options} as:
                                Option_A: Probability_A
                                Option_B: Probability_B
                                ...
                                Option_N: Probability_N
            
                                """
            )
            model_name = "claude3.7"

        elif notepad.num_predictions_attempted > 4:
            prompt = clean_indents(
                f"""
                        You are an advanced superintelligent artificial general intelligence (AGI) used to make accurate forecasts of the future. Your accuracy is being tested using the following formula for scoring: 
                        {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                        Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 5% and 95% unless absolutely sure.

                        Today is {datetime.now().strftime("%Y-%m-%d")}.

                        The following question is to be predicted:
                       {question.question_text}
                        The options are: {question.options}
                        {question.resolution_criteria}
                        {question.fine_print}

                       The following additional information on the subject has been provided by your engineers:
                       {question.background_info}

                       Another AI has done the following research using news available online.
                       {research}

                       Before providing a final forecast, provide the following:
                       (0) How many days are left until the question resolution. 
                       (1) Using a top-down approach list factors you have to consider and key facts.
                       (2a) Reasons the question might resolve as No.
                       (2b) Reasons the question might resolve as Yes.
                       (3) Based on these reasons what would a resolution probability be?
                       (4a) Reflect on this answer and the considerations above.
                       (4b) Adjust your opinion accordingly.

                       You write your prediction including the reasoning for your answer.

                       The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            
                        """
            )
            model_name = "gpt4o"

        reasoning = await self.get_llm(model_name, "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        notepad = await self._get_notepad(question)
        model_name = "default"
        if notepad.num_predictions_attempted == 1:
            prompt = clean_indents(
                f"""
                        You are a professional forecaster interviewing for a job.

                        Your interview question is:
                        {question.question_text}

                        Question background:
                        {question.background_info}

                        This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
                        {question.resolution_criteria}

                        {question.fine_print}


                        An AI aiding your research provides the following:
                        {research}

                        Today is {datetime.now().strftime("%Y-%m-%d")}.
                        
                        Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
                        {lower_bound_message}
                        {upper_bound_message}
                        
                        Formatting Instructions:
                        - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
                        - Never use scientific notation.
                        - Always start with a smaller number (more negative if negative) and then increase from there
                        
                        Before answering you write:
                        (a) The time left until the outcome to the question is known.
                        (b) The current value and current rate of change (whether it changes linearly, exponentially, is normally distributed or other)
                        (c) What the value at the resolution time will be if the current rate of change remains unchanged.
                        (d) Factors that might change the rate of change (extreme outliers).

                        Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 1% and 99%. Remember to not be overconfident.
                        You write your rationale remembering that good forecasters put extra weight on the status quo outcome since the world changes slowly most of the time.

                        The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
                        """
            )
            model_name = "claude3.7"
        elif notepad.num_predictions_attempted == 2:
            prompt = clean_indents(
                f"""
                You are an economics researcher working on a model for predicting future events. You hold a capitalistic word view and believe in the efficient-market hypothesis.

                Your are currently trying to find the chance of this happening:
                {question.question_text}

                You have already compiled additional information to aid you in modelling this event:
                {question.background_info}

                To test the accuracy of your forecast a specific condition has been chosen to evaluate the future outcome of the event:
                {question.resolution_criteria}

                {question.fine_print}

                Your research assistant says:
                {research}

                Today is {datetime.now().strftime("%Y-%m-%d")}.
                Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
                {lower_bound_message}
                {upper_bound_message}
                
                Formatting Instructions:
                - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
                - Never use scientific notation.
                - Always start with a smaller number (more negative if negative) and then increase from there
                Before answering you write:
                Before answering you write:
                        (a) The time left until the outcome to the question is known.
                        (b) The current value and current rate of change (whether it changes linearly, exponentially, is normally distributed or other)
                        (c) What the value at the resolution time will be if the current rate of change remains unchanged.
                        (d) Factors that might change the rate of change (extreme outliers).

                A log score is used to evaluate your performance. That means, more extreme forecasts will only result in a marginally better score when resolved correctly, but a significantly worse score otherwise. Thus don't be overconfident and leave long tails.

                The last thing you write is your final answer as:
                "
                Percentile 10: XX
                Percentile 20: XX
                Percentile 40: XX
                Percentile 60: XX
                Percentile 80: XX
                Percentile 90: XX
                "
                                """
            )
            model_name = "o1preview"  # ToDo: o1preview

        elif notepad.num_predictions_attempted == 3:
            prompt = clean_indents(
                f"""
               You are a journalist writing a piece on prediction markets. To learn more about forecasting you are taking part in a forecasting tournament.
               Your are currently trying to find the chance of this happening:
               {question.question_text}

               You know the following additional information on the subject :
               {question.background_info}

               To create a fair environment a specific criteria has been chosen to evaluate the question's outcome:
               {question.resolution_criteria}

               {question.fine_print}

               Your online research has revealed the following:
               {research}

               Today is {datetime.now().strftime("%Y-%m-%d")}.
                Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
                {lower_bound_message}
                {upper_bound_message}
                
                Formatting Instructions:
                - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
                - Never use scientific notation.
                - Always start with a smaller number (more negative if negative) and then increase from there
               Before answering you write:
               Before answering you write:
                        (a) The time left until the outcome to the question is known.
                        (b) The current value and current rate of change (whether it changes linearly, exponentially, is normally distributed or other)
                        (c) What the value at the resolution time will be if the current rate of change remains unchanged.
                        (d) Factors that might change the rate of change (extreme outliers).

               You write your rationale thinking about black swan events, that, while extremely unlikely, can have huge impacts. A log score is used to evaluate your performance. That means, e.g. predicting 0.5% when the event is 5% likely to occur, will only result in a marginally better score when this question resolves "No", but a significantly worse score if the question resolves "Yes". Don't go towards extreme answers.

               The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
                                   """
            )
            model_name = "gpt4o"

        elif notepad.num_predictions_attempted == 4:
            prompt = clean_indents(
                f"""
                You are an advanced superintelligent artificial general intelligence (AGI) used to make accurate forecasts of the future. Your accuracy is being tested using the following formula for scoring: 
                {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 5% and 95% unless absolutely sure.

                Today is {datetime.now().strftime("%Y-%m-%d")}.

                The following question is to be predicted:
                {question.question_text}

                {question.resolution_criteria}
                {question.fine_print}

                The following additional information on the subject has been provided by your engineers:
                {question.background_info}

                Another AI has done the following research using news available online.
                {research}
                Today is {datetime.now().strftime("%Y-%m-%d")}.
                Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
                {lower_bound_message}
                {upper_bound_message}
                
                Formatting Instructions:
                - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
                - Never use scientific notation.
                - Always start with a smaller number (more negative if negative) and then increase from there
                Before providing a final forecast, provide the following:
                Before answering you write:
                (a) The time left until the outcome to the question is known.
                (b) The current value and current rate of change (whether it changes linearly, exponentially, is normally distributed or other)
                (c) What the value at the resolution time will be if the current rate of change remains unchanged.
                (d) Factors that might change the rate of change (extreme outliers).
                (e) Reflect on the two points above.
                (f) Adjust your opinion accordingly.

                You write your prediction including the reasoning for your answer.

                The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
                                """
            )
            model_name = "o1preview"

        elif notepad.num_predictions_attempted == 5:
            comments = "\n".join(notepad.note_entries["reasonings"])
            prompt = clean_indents(
                f"""
                                You are a superforecaster aggregating other users' and experts' comments to arrive at a chance of future events occuring. A forecasting tournament is using the following formula for scoring: 
                                {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                                This means forecasts at the extreme ends provide decreasing marginal beneftis, while leading to a exponentially increasing point loss if predicted incorrectly. As such good forecasters usually predict between 1% and 99% unless the resolution is already apparent (e.g. if the status quo is impossible to change in the remaining time). 
                                 You want to win the competition and get the highest score.
                                 You are trying to find the chance of this happening:
                                {question.question_text}

                                Additional information has been provided to aid in forecasting:
                                {question.background_info}

                                To create a fair environment a specific criteria has been chosen to evaluate the question's outcome:
                                {question.resolution_criteria}
                                Be mindful of these stipulations, some of which may be unintuitive to the way the question is worded, but take care of edge cases:
                                {question.fine_print}

                                You have researched using online news.
                                {research}
                                You are looking at the comments from other users, experts and laypeople alike. Use their combined forecast for your prediction.
                                {comments}
                                Today is {datetime.now().strftime("%Y-%m-%d")}.
                                    Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
                        {lower_bound_message}
                        {upper_bound_message}
                        
                        Formatting Instructions:
                        - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
                        - Never use scientific notation.
                        - Always start with a smaller number (more negative if negative) and then increase from there
                        Look through each comment and comprehend their trains of thought. 
                        
                        Before assigning weights write the following:
                        Before answering you write:
                        (a) The time left until the outcome to the question is known.
                        (b) The current value and current rate of change (whether it changes linearly, exponentially, is normally distributed or other)
                        (c) What the value at the resolution time will be if the current rate of change remains unchanged.
                        (d) Factors that might change the rate of change (extreme outliers).

                        The last thing you write is your final answer as:
                        "
                        Percentile 10: XX
                        Percentile 20: XX
                        Percentile 40: XX
                        Percentile 60: XX
                        Percentile 80: XX
                        Percentile 90: XX
                        "
                        """
            )
            model_name = "claude3.7"

        elif notepad.num_predictions_attempted > 4:
            prompt = clean_indents(
                f"""
                        You are an advanced superintelligent artificial general intelligence (AGI) used to make accurate forecasts of the future. Your accuracy is being tested using the following formula for scoring: 
                        {chr(92)}frac{{100}}{{n}} {chr(92)}sum_{{i=1}}^{{N}} {chr(92)}log {chr(92)}text{{score}}(p) - {chr(92)}log {chr(92)}text{{score}}(p_i)
                        Forecasts at the extreme ends provide marginal benefits, while leading to a massive point loss if predicted incorrectly. As such, keep forecasts between 5% and 95% unless absolutely sure.

                        Today is {datetime.now().strftime("%Y-%m-%d")}.

                        The following question is to be predicted:
                       {question.question_text}

                        {question.resolution_criteria}
                        {question.fine_print}

                       The following additional information on the subject has been provided by your engineers:
                       {question.background_info}

                       Another AI has done the following research using news available online.
                       {research}
                                    Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}
                        {lower_bound_message}
                        {upper_bound_message}
                        
                        Formatting Instructions:
                        - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
                        - Never use scientific notation.
                        - Always start with a smaller number (more negative if negative) and then increase from there
                       Before providing a final forecast, provide the following:
                       Before answering you write:
                        (a) The time left until the outcome to the question is known.
                        (b) The current value and current rate of change (whether it changes linearly, exponentially, is normally distributed or other)
                        (c) What the value at the resolution time will be if the current rate of change remains unchanged.
                        (d) Factors that might change the rate of change (extreme outliers).

                       You write your prediction including the reasoning for your answer.

                       The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
                        """
            )
            model_name = "default"

        reasoning = await self.get_llm(model_name, "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = Q2TemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="metaculus/anthropic/claude-3-7-sonnet-latest",
                temperature=1,
                timeout=40,
                allowed_tries=2,
            ),

            "claude3.7": GeneralLlm(
                model="metaculus/anthropic/claude-3-7-sonnet-latest",
                temperature=0.7,
                timeout=40,
                allowed_tries=2,
            ),
            "o1preview": GeneralLlm(
                model="metaculus/openai/o1",
                temperature=0.5,
                timeout=40,
                allowed_tries=2,
            ),
            "gpt4o": GeneralLlm(
                model="metaculus/openai/gpt-4o",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),

            "summarizer": "openai/gpt-4o-mini",
        }
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = False
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            #"https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    template_bot.log_report_summary(forecast_reports)  # type: ignore