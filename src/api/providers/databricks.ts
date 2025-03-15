import axios from "axios"
import { ApiStream } from "../transform/stream"
import { BaseProvider } from "./base-provider"
import { ApiHandlerOptions, ModelInfo } from "../../shared/api"
import { Anthropic } from "@anthropic-ai/sdk"
import { ANTHROPIC_DEFAULT_MAX_TOKENS } from "./constants"

interface DatabricksResponse {
	choices: Array<{
		message: {
			content: string
		}
	}>
	usage?: {
		prompt_tokens: number
		completion_tokens: number
	}
}

export class DatabricksProvider extends BaseProvider {
	private options: ApiHandlerOptions
	private baseUrl: string
	private token: string
	private modelId: string

	constructor(options: ApiHandlerOptions) {
		super()
		this.options = options
		this.baseUrl = options.databricksBaseUrl || ""
		this.token = options.databricksApiKey || ""
		this.modelId = options.databricksModelId || "databricks-meta-llama-3-1-8b-instruct"
	}

	async *createMessage(systemPrompt: string, messages: Anthropic.Messages.MessageParam[]): ApiStream {
		if (!this.baseUrl) {
			throw new Error("Databricks base URL is required")
		}
		if (!this.token) {
			throw new Error("Databricks API token is required")
		}

		const { maxTokens, temperature } = this.getModel()

		const requestData = {
			messages: [
				{
					role: "system",
					content: systemPrompt,
				},
				...messages.map((msg) => ({
					role: msg.role,
					content: Array.isArray(msg.content)
						? msg.content.map((block) => {
							if (block.type === "text") return block.text
							if (block.type === "image" && block.source?.type === "base64") {
								return `[Image: ${block.source.media_type}]`
							}
							return ""
						}).join("\n")
						: msg.content,
				})),
			],
			max_tokens: maxTokens ?? ANTHROPIC_DEFAULT_MAX_TOKENS,
			temperature: temperature ?? 0.7,
		}

		try {
			const response = await axios.post<DatabricksResponse>(
				`${this.baseUrl}/serving-endpoints/${this.modelId}/invocations`,
				requestData,
				{
					headers: {
						"Content-Type": "application/json",
						Authorization: `Bearer ${this.token}`,
					},
				}
			)

			// Yield usage information if available
			if (response.data.usage) {
				yield {
					type: "usage",
					inputTokens: response.data.usage.prompt_tokens,
					outputTokens: response.data.usage.completion_tokens,
				}
			}

			// Yield the response text
			const content = response.data.choices[0]?.message?.content
			if (content) {
				yield { type: "text", text: content }
			} else {
				throw new Error("Invalid response format from Databricks API")
			}
		} catch (error) {
			console.error("Databricks API Error:", error)
			if (axios.isAxiosError(error)) {
				throw new Error(
					`Databricks API error: ${error.response?.data?.message || error.message}`
				)
			}
			throw error
		}
	}

	getModel(): { id: string; info: ModelInfo; maxTokens?: number; temperature?: number } {
		const info: ModelInfo = {
			maxTokens: 4096,
			contextWindow: 8192,
			supportsImages: false,
			supportsPromptCache: false,
			supportsComputerUse: false,
			inputPrice: 0, // Update with actual pricing when available
			outputPrice: 0, // Update with actual pricing when available
			description: `Databricks model: ${this.modelId}`,
		}

		return {
			id: this.modelId,
			info,
			maxTokens: this.options.modelMaxTokens,
			temperature: this.options.modelTemperature ?? 0.7,
		}
	}

	/**
	 * Override token counting if Databricks provides a token counting endpoint
	 * Otherwise fall back to tiktoken implementation from BaseProvider
	 */
	override async countTokens(
		content: Array<Anthropic.Messages.ContentBlockParam>
	): Promise<number> {
		try {
			// TODO: Implement Databricks token counting endpoint if available
			return super.countTokens(content)
		} catch (error) {
			console.warn("Databricks token counting failed, using fallback", error)
			return super.countTokens(content)
		}
	}
}