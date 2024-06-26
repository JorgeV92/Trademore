open Trademore.Alpaca

module TypeStock = struct 
  
    type industry = Technology | Healthcare | Finance | ConsumerGoods | Utilites
    type market_cap = Large | Mid | Small
    type risk_level = High | Medium | Low

    type stock = {
      name : string;
      ticker : string;
      industry : industry;
      market_cap : market_cap;
      risk : risk_level;
    }

    let create_stock name ticker industry market_cap risk =  {
      namel ticker; industry; market_cap; risk;
    }

    let filter_by_industry_ stocks industry = 
      List.filter (fun s -> s.industry = industry) stocks 

    (* Function that only applies to large-cap tech stocks *)
    let invest_in_large_tech stocks =
      stocks 
      |> List.filter (fun s -> s.industry = Technology && s.market_cap = Large)
      |> List.map (fun s -> (* TODO *))
end